import os
import re
import logging
import numpy as np
import tensorflow as tf
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model.train import StyleSynthesisModel, StyleDataReader
import tempfile
import model.drawing as drawing
import matplotlib
# Use non-interactive backend for better performance in server environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading
import concurrent.futures
from functools import partial
import requests
import io
from PIL import Image
from utils.models import MarkdownRequest, WordRequest
from utils.markdown_parser import parse_markdown
from utils.strokes_processor import (
    metadata_to_style,
    process_single_item,
    process_batch_items,
    combine_segments_as_one_line,
    calculate_indentation,
    calculate_dynamic_spacing,
    ensure_bounds_preserve_spacing,
    ensure_bounds
)

# Add environment variable for handwriting quality service URL
GEMINI_SERVICE_URL = os.getenv('GEMINI_SERVICE_URL', 'http://handwriting-quality:5000')

# But without complex multiprocessing that causes session issues
cpu_count = os.cpu_count() or 4
config = tf.ConfigProto(
    intra_op_parallelism_threads=cpu_count,  # Use all available CPU cores
    inter_op_parallelism_threads=cpu_count,  # Use all available CPU cores
    allow_soft_placement=True,
    # Enable memory growth to prevent GPU memory issues
    gpu_options=tf.GPUOptions(allow_growth=True)
)

# Use a single global session that can be shared across threads (but not processes)
session = tf.Session(config=config)
tf.keras.backend.set_session(session)

# Set seeds for determinism
random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)

app = FastAPI()

# Allow requests from your HTML server
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # Adjust this list as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# Global thread lock for TensorFlow operations
tf_lock = threading.RLock()

# Utility functions
SMART_QUOTES = {""": '"', """: '"', "'": "'", "'": "'", "–": '-', "—": '-'}
HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.*)")
BULLET_PATTERN = re.compile(r"^(\s*)-\s+(.*)")  # Capture leading spaces for nesting
NUMBERED_PATTERN = re.compile(r"^(\s*)\d+\.\s+(.*)")  # Capture leading spaces for nesting
BLOCKQUOTE_PATTERN = re.compile(r"^(\s*)>\s+(.*)")  # Capture leading spaces for nesting
EMPHASIS_BOLD = re.compile(r"(\*\*|__)")
EMPHASIS_ITALIC = re.compile(r"(\*|_)")


@app.get("/health")
def health():
    region = os.environ.get('CLOUD_RUN_REGION', 'unknown')
    vm_name = os.environ.get('HOSTNAME', 'unknown')
    return {"message": "OK", "region": region, "instance": os.environ.get('K_REVISION', 'unknown'), "vm": vm_name}

@app.get("/hello")
def hello():
    return {"message": "Hello, World!"}

@app.post("/convert")
def convert_markdown(request: MarkdownRequest):
    """Convert markdown text to handwritten strokes.

    This endpoint takes markdown text and converts it into a sequence of handwritten strokes.
    It supports both default style-based generation and reference stroke-based generation.
    The endpoint processes the markdown text line by line, maintaining formatting and structure.

    Args:
        request (MarkdownRequest): The request object containing:
            - markdown (str): The markdown text to convert
            - style_id (int, optional): The ID of the handwriting style to use. Defaults to 8.
            - ref_strokes (list, optional): Reference strokes to guide the handwriting style.
                If provided, these strokes will be used to influence the generated handwriting.
            - screen_width (int, optional): The width of the screen. Defaults to 800.
            - screen_height (int, optional): The height of the screen. Defaults to 600.

    Returns:
        dict: A dictionary containing:
            - strokes (list): List of stroke groups, where each group contains:
                - line (str): The original text line
                - strokes (list): List of stroke sequences for the line
                - stroke_width (int): Width of the strokes
                - stroke_color (str): Color of the strokes

    Raises:
        HTTPException: If there's an error processing the style metadata or other processing errors.

    Notes:
        - The endpoint supports parallel processing of lines for better performance
        - If reference strokes are provided, a quality check is performed on first 5 words only
        - If the quality check fails, it falls back to default style generation
        - The endpoint maintains markdown formatting like headers, lists, and indentation
    """
    # 1) Parse markdown → lines + metadata
    try:
        logging.info(f"Processing markdown: {request}")
        lines, metadata = parse_markdown(request.markdown)
        joined = "\n".join(lines)
        styles, biases, stroke_colors, stroke_widths = metadata_to_style(
            metadata,
            style_id=request.style_id,
            lines=joined
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Style metadata error: {e}")

    # 2) Build per-line items
    indexed_items = []
    for i, (line, meta, bias, style, sw, sc) in enumerate(
        zip(lines, metadata, biases, styles, stroke_widths, stroke_colors)
    ):
        indexed_items.append({
            "index": i,
            "line": line,
            "metadata": meta,
            "bias": bias,
            "style": style,
            "stroke_width": sw,
            "stroke_color": sc
        })

    # Check if reference style strokes were provided
    ref_strokes = request.ref_strokes
    screen_width = request.screen_width
    screen_height = request.screen_height
    
    # If we have ref_strokes, perform quality check on first 5 words only
    if ref_strokes:
        logging.info("Reference strokes provided, performing quality check on first 5 words")
        
        # Extract first 5 words from the text for quality checking
        all_text = " ".join([item["line"] for item in indexed_items if item["line"].strip()])
        words = all_text.split()
        first_5_words = " ".join(words[:5]) if words else ""
        
        if first_5_words:
            try:
                # Create a minimal test item with first 5 words
                test_item = {
                    "index": 0,
                    "line": first_5_words,
                    "metadata": {"type": "paragraph", "indent": 0, "font_scale": 1.0},
                    "bias": 0.75,
                    "style": request.style_id,
                    "stroke_width": 1,
                    "stroke_color": "black"
                }
                
                # Process only the test item
                logging.info(f"Processing test phrase for quality check: '{first_5_words}'")
                test_result = process_single_item(test_item, ref_strokes, screen_width, screen_height)
                
                # Create temporary image for quality check
                temp_path = None
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Save test strokes to image
                test_strokes = test_result["strokes"]
                if test_strokes:
                    logging.info(f"Saving {len(test_strokes)} test stroke groups for quality check")
                    save_strokes_to_image(test_strokes, temp_path)
                    
                    # Check handwriting quality
                    quality_check = check_handwriting_quality(temp_path)
                    
                    # Clean up the temporary file
                    try:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            logging.debug(f"Cleaned up temporary file: {temp_path}")
                    except Exception as e:
                        logging.error(f"Error cleaning up temporary file {temp_path}: {e}")
                    
                    # If quality check fails, regenerate with default style (prevent recursion)
                    if not quality_check:
                        logging.info("Handwriting quality check FAILED on test phrase, regenerating with default style")
                        fallback_request = MarkdownRequest(
                            markdown=request.markdown,
                            style_id=request.style_id,
                            ref_strokes=None,  # Remove ref_strokes to prevent recursion
                            screen_width=request.screen_width,
                            screen_height=request.screen_height
                        )
                        return convert_markdown(fallback_request)
                    else:
                        logging.info("Handwriting quality check PASSED on test phrase, proceeding with full generation")
                else:
                    logging.warning("No test strokes generated, proceeding with full generation")
                    
            except Exception as e:
                logging.error(f"Error during test phrase quality check: {e}")
                logging.info("Falling back to default style due to quality check error")
                fallback_request = MarkdownRequest(
                    markdown=request.markdown,
                    style_id=request.style_id,
                    ref_strokes=None,  # Remove ref_strokes to prevent recursion
                    screen_width=request.screen_width,
                    screen_height=request.screen_height
                )
                return convert_markdown(fallback_request)
        else:
            logging.warning("No words found for quality check, proceeding with full generation")

    # Use screen dimensions from the request to set responsive line spacing
    screen_width = request.screen_width
    screen_height = request.screen_height

    # Responsive line spacing: 10% of screen height, min 30, max 60
    line_spacing = max(30, min(screen_height * 0.10, 60))
    
    # Determine optimal number of threads based on CPU count
    cpu_count = os.cpu_count() or 4
    # Reduce worker threads to prevent over-subscription and improve performance
    # Since we have a thread lock bottleneck, more threads won't help
    max_workers = min(cpu_count, 8)  # Cap at 8 to prevent overhead
    
    logging.info(f"Using {max_workers} worker threads with {cpu_count} CPU cores")
    
    # Process items in batches for better performance
    batch_size = 4  # Process 4 items at a time to balance memory and performance
    processed_lines = []
    
    try:
        # Process items in batches
        for i in range(0, len(indexed_items), batch_size):
            batch_items = indexed_items[i:i + batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{(len(indexed_items) + batch_size - 1)//batch_size}")
            
            batch_results = process_batch_items(batch_items, ref_strokes, screen_width, screen_height)
            processed_lines.extend(batch_results)
            
    except Exception as e:
        # Fall back to individual processing if batch processing fails
        logging.error(f"Batch processing failed, falling back to individual: {e}")
        processed_lines = []
        for item in indexed_items:
            try:
                processed = process_single_item(item, ref_strokes, screen_width, screen_height)
                processed_lines.append(processed)
            except Exception as item_e:
                logging.error(f"Individual processing failed for item {item['index']}: {item_e}")
                processed_lines.append({
                    "index": item["index"],
                    "line": item["line"],
                    "strokes": [],
                    "stroke_width": item["stroke_width"],
                    "stroke_color": item["stroke_color"],
                    "metadata": item["metadata"]
                })
    
    # Sort the results by index to maintain original order
    processed_lines.sort(key=lambda x: x["index"])

    # Merge per-line results into full-page output
    grouped = {}
    for entry in sorted(processed_lines, key=lambda x: x["index"]):
        gid = entry["metadata"]["group_id"]
        grp = grouped.setdefault(gid, {
            "lines": [], "segments": [],
            "stroke_width": entry["stroke_width"],
            "stroke_color": entry["stroke_color"],
            "metadata": entry["metadata"]
        })
        grp["lines"].append(entry["line"])
        grp["segments"].append(entry["strokes"])

    merged_output = []
    # Start with a small top margin. This will track the "bottom" of the rendered content.
    cumulative_y_offset = 20.0

    for gid in sorted(grouped):
        group = grouped[gid]
        text = " ".join(filter(None, group["lines"]))
        combined = combine_segments_as_one_line(group["segments"], screen_width, screen_height)

        group_metadata = group["metadata"]

        # Calculate indentation and apply it
        indent_offset = calculate_indentation(group_metadata, screen_width)
        if indent_offset > 0:
            for stroke in combined:
                for pt in stroke:
                    pt[0] += indent_offset

        # Calculate the vertical gap needed before this line.
        spacing = calculate_dynamic_spacing(
            group_metadata, screen_width, screen_height, line_spacing
        )

        # For empty lines (which have no strokes), just add the spacing gap and move on.
        if not combined:
            if group_metadata.get("type") == "empty":
                cumulative_y_offset += spacing
            continue

        # Find the bounding box of the strokes to get their true top and height.
        all_y = [pt[1] for stroke in combined for pt in stroke]
        min_y, max_y = min(all_y), max(all_y)
        height_of_group = max_y - min_y
        
        # The new top position is the previous bottom plus the calculated spacing.
        target_y = cumulative_y_offset + spacing
        
        # The amount to shift is the difference between the target top (target_y)
        # and the strokes' current top (min_y).
        y_shift = target_y - min_y
        
        # Apply the vertical shift to all points in the group.
        for stroke in combined:
            for pt in stroke:
                pt[1] += y_shift
        
        # Update the cumulative offset to the new bottom of the content.
        cumulative_y_offset = target_y + height_of_group

        merged_output.append({
            "line": text,
            "strokes": combined,
            "stroke_width": group["stroke_width"],
            "stroke_color": group["stroke_color"],
            "metadata": group_metadata  # Include metadata in output for debugging
        })

    # Apply final bounds checking that preserves line spacing
    merged_output = ensure_bounds_preserve_spacing(merged_output, screen_width, screen_height)

    # Return the generated strokes
    logging.info("Returning generated strokes")
    return {"strokes": merged_output}

@app.post("/convert-word")
def convert_word(request: WordRequest):
    """Convert a word or two words to handwritten strokes.

    This endpoint takes simple text (a word or two words) and converts it into handwritten strokes.
    It's designed for simple, fast word-level handwriting generation without complex formatting.

    Args:
        request (WordRequest): The request object containing:
            - text (str): The word or words to convert (e.g., "hello" or "hello world")
            - style_id (int, optional): The ID of the handwriting style to use. Defaults to 8.
            - ref_strokes (list, optional): Reference strokes to guide the handwriting style.

    Returns:
        dict: A dictionary containing:
            - strokes (list): List of stroke sequences for the word(s)
            - text (str): The original input text

    Raises:
        HTTPException: If there's an error processing the text or generating strokes.
    """
    try:
        logging.info(f"Processing word request: {request.text}")
        
        # Validate input
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Clean and normalize the text
        text = request.text.strip()
        
        # Check if reference style strokes were provided
        ref_strokes = request.ref_strokes
        
        # If we have ref_strokes, perform quality check
        if ref_strokes:
            logging.info("Reference strokes provided, performing quality check")
            
            try:
                # Create a test item for quality checking
                test_item = {
                    "index": 0,
                    "line": text,
                    "metadata": {
                        "type": "paragraph", 
                        "indent": 0, 
                        "font_scale": 1.0,
                        "spacing_before": 0.5,
                        "spacing_after": 0.5
                    },
                    "bias": 0.75,
                    "style": request.style_id,
                    "stroke_width": 1,
                    "stroke_color": "black"
                }
                
                # Process test item with reference strokes
                logging.info(f"Processing test word for quality check: '{text}'")
                test_result = process_single_item(test_item, ref_strokes, 800, 600)
                
                # Create temporary image for quality check
                temp_path = None
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Save test strokes to image
                test_strokes = test_result["strokes"]
                if test_strokes:
                    logging.info(f"Saving {len(test_strokes)} test stroke groups for quality check")
                    save_strokes_to_image(test_strokes, temp_path)
                    
                    # Check handwriting quality
                    quality_check = check_handwriting_quality(temp_path)
                    
                    # Clean up the temporary file
                    try:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            logging.debug(f"Cleaned up temporary file: {temp_path}")
                    except Exception as e:
                        logging.error(f"Error cleaning up temporary file {temp_path}: {e}")
                    
                    # If quality check fails, regenerate with default style (prevent recursion)
                    if not quality_check:
                        logging.info("Handwriting quality check FAILED on test word, regenerating with default style")
                        fallback_request = WordRequest(
                            text=request.text,
                            style_id=request.style_id,
                            ref_strokes=None  # Remove ref_strokes to prevent recursion
                        )
                        return convert_word(fallback_request)
                    else:
                        logging.info("Handwriting quality check PASSED on test word, proceeding with full generation")
                else:
                    logging.warning("No test strokes generated, proceeding with full generation")
                    
            except Exception as e:
                logging.error(f"Error during test word quality check: {e}")
                logging.info("Falling back to default style due to quality check error")
                fallback_request = WordRequest(
                    text=request.text,
                    style_id=request.style_id,
                    ref_strokes=None  # Remove ref_strokes to prevent recursion
                )
                return convert_word(fallback_request)
        
        # Create a simple item for word processing
        item = {
            "index": 0,
            "line": text,
            "metadata": {
                "type": "paragraph", 
                "indent": 0, 
                "font_scale": 1.0,
                "spacing_before": 0.5,
                "spacing_after": 0.5
            },
            "bias": 0.75,  # Good balance for word-level generation
            "style": request.style_id,
            "stroke_width": 1,
            "stroke_color": "black"
        }
        
        # Process the word using the existing single item processor
        result = process_single_item(
            item, 
            ref_strokes=request.ref_strokes,
            screen_width=800,  # Fixed width for word-level API
            screen_height=600  # Fixed height for word-level API
        )
        
        # Extract just the strokes from the result
        strokes = result.get("strokes", [])
        
        # Optimize and clean up the strokes
        if strokes:
            # Apply bounds checking for word-level output
            strokes = ensure_bounds(strokes, 800, 600, margin=20)
            
            # Optimize stroke precision
            for stroke in strokes:
                for point in stroke:
                    point[0] = round(point[0], 2)
                    point[1] = round(point[1], 2)
        
        logging.info(f"Successfully generated strokes for word: '{text}'")
        
        return {
            "text": text,
            "strokes": strokes,
            "style_id": request.style_id
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logging.error(f"Error processing word request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing word: {str(e)}")

def save_strokes_to_image(strokes, output_path):
    """
    Saves the strokes to an image file with proper error handling.
    """
    try:
        # Validate input
        if not strokes:
            logging.warning("No strokes provided to save_strokes_to_image")
            # Create a blank image
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')
            ax.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300, format='png')
            plt.close(fig)
            return
        
        # Create figure with white background
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Plot each stroke
        stroke_count = 0
        for stroke in strokes:
            if len(stroke) < 2:
                continue
            xs = [pt[0] for pt in stroke]
            ys = [pt[1] for pt in stroke]
            ax.plot(xs, ys, color='black', linewidth=1)
            stroke_count += 1
        
        logging.info(f"Rendered {stroke_count} strokes to image")
        
        # Remove axes and set equal aspect
        ax.axis('off')
        ax.set_aspect('equal')
        
        # Save the figure as PNG
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300, format='png')
        plt.close(fig)
        
        # Verify the file was created and has content
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logging.info(f"Successfully saved image to {output_path} (size: {file_size} bytes)")
        else:
            raise RuntimeError(f"Failed to create image file at {output_path}")
            
    except Exception as e:
        logging.error(f"Error saving strokes to image: {e}")
        # Clean up any partial file
        try:
            if os.path.exists(output_path):
                os.unlink(output_path)
        except:
            pass
        raise

def check_handwriting_quality(image_path):
    """
    Calls the handwriting quality check API with improved error handling and validation.
    """
    # Validate input
    if not image_path or not os.path.exists(image_path):
        logging.error(f"Image file does not exist: {image_path}")
        return True  # Default to accepting if we can't validate
    
    # Check file size
    try:
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            logging.error(f"Image file is empty: {image_path}")
            return True  # Default to accepting empty files
        logging.info(f"Checking handwriting quality for image: {image_path} (size: {file_size} bytes)")
    except Exception as e:
        logging.error(f"Error checking image file size: {e}")
        return True
    
    try:
        # Test service availability first
        health_response = requests.get(f"{GEMINI_SERVICE_URL}/health", timeout=2)
        if health_response.status_code != 200:
            logging.warning(f"Handwriting quality service health check failed: {health_response.status_code}")
            return True  # Default to accepting if service is unhealthy
        
        # Send the image for quality check
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(
                f"{GEMINI_SERVICE_URL}/check_handwriting", 
                files=files, 
                timeout=10  # Increased timeout for AI processing
            )
            response.raise_for_status()
            result = response.json()
            
            # Validate response structure
            if not isinstance(result, dict):
                logging.error(f"Invalid response format from quality service: {result}")
                return True
            
            ok = result.get('ok', False)
            quality_result = result.get('result', False)
            
            if not ok:
                error_msg = result.get('error', 'Unknown error')
                logging.warning(f"Quality check service returned error: {error_msg}")
                return True  # Default to accepting on service errors
            
            logging.info(f"Quality check result: {quality_result}")
            return quality_result
            
    except requests.exceptions.Timeout:
        logging.warning("Handwriting quality service request timed out")
        return True  # Default to accepting on timeout
    except requests.exceptions.ConnectionError:
        logging.warning("Cannot connect to handwriting quality service")
        return True  # Default to accepting if service is unavailable
    except requests.exceptions.RequestException as e:
        logging.warning(f"Handwriting quality service request failed: {e}")
        return True  # Default to accepting on request errors
    except Exception as e:
        logging.error(f"Unexpected error checking handwriting quality: {e}")
        return True  # Default to accepting on unexpected errors

if __name__ == '__main__':
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    port = int(os.getenv('PORT', 8080))
    logging.info(f"Starting application on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")