import os
import re
import logging
import numpy as np
import tensorflow as tf
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from train import StyleSynthesisModel, StyleDataReader
import tempfile
import uuid
import drawing
import matplotlib
# Use non-interactive backend for better performance in server environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading
import concurrent.futures
from functools import partial
import json
from datetime import datetime
import requests
import io
from PIL import Image

# Add environment variable for handwriting quality service URL
GEMINI_SERVICE_URL = os.getenv('GEMINI_SERVICE_URL', 'http://handwriting-quality:5000')

# But without complex multiprocessing that causes session issues
config = tf.ConfigProto(
    intra_op_parallelism_threads=4,  # Use fixed value
    inter_op_parallelism_threads=4,  # Use fixed value
    allow_soft_placement=True
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

class MarkdownRequest(BaseModel):
    markdown: str
    style_id: Optional[int] = 8  # Default style id if not provided
    ref_strokes: Optional[list] = None
    screen_width: Optional[int] = 800  # Default screen width
    screen_height: Optional[int] = 600  # Default screen height

# Set up the model
style_model = StyleSynthesisModel(
    log_dir="logs_style_synthesis",
    checkpoint_dir="checkpoints_style_synthesis",
    prediction_dir="predictions_style_synthesis",
    learning_rates=[1e-4],
    batch_sizes=[32],
    patiences=[2000],
    beta1_decays=[0.9],
    validation_batch_size=32,
    optimizer='rms',
    num_training_steps=30000,
    regularization_constant=0.0,
    keep_prob=1.0,
    enable_parameter_averaging=False,
    min_steps_to_checkpoint=2000,
    log_interval=50,
    grad_clip=10,
    lstm_size=400,
    output_mixture_components=20,
    attention_mixture_components=10,
    style_embedding_size=256
)

# Restore from latest or specific checkpoint
checkpoint_path = "final_checkpoints/model-10350"
style_model.saver.restore(style_model.session, checkpoint_path)

# Set deterministic sampling
style_model.sampling_mode = "deterministic"
style_model.force_deterministic_sampling = True
style_model.temperature = 0.5

# Global thread lock for TensorFlow operations
tf_lock = threading.RLock()

# Utility functions
SMART_QUOTES = {""": '"', """: '"', "'": "'", "'": "'", "–": '-', "—": '-'}
HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.*)")
BULLET_PATTERN = re.compile(r"^-\s+(.*)")
NUMBERED_PATTERN = re.compile(r"^\d+\.\s+(.*)")
BLOCKQUOTE_PATTERN = re.compile(r"^>\s+(.*)")
EMPHASIS_BOLD = re.compile(r"(\*\*|__)")
EMPHASIS_ITALIC = re.compile(r"(\*|_)")

def parse_markdown(markdown_text):
    """
    Parses a markdown string and returns:
      - a list of text lines (with markdown markers removed and normalized)
      - a list of metadata dictionaries corresponding to each line.
    """
    SMART_QUOTES = {
        """: '"', """: '"', "'": "'",
        "'": "'", "–": "-", "—": "-"
    }
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.*)")
    BULLET_PATTERN = re.compile(r"^-\s+(.*)")
    NUMBERED_PATTERN = re.compile(r"^\d+\.\s+(.*)")
    BLOCKQUOTE_PATTERN = re.compile(r"^>\s+(.*)")
    EMPHASIS_BOLD = re.compile(r"(\*\*|__)")
    EMPHASIS_ITALIC = re.compile(r"(\*|_)")

    results = []
    line_meta_base = {"type": "paragraph", "indent": 0}

    for line_idx, raw_line in enumerate(markdown_text.splitlines()):
        line = raw_line.strip()
        line_meta = line_meta_base.copy()

        # Assign each original raw line an ID
        line_meta["group_id"] = line_idx

        # Detect markdown constructs
        if line.startswith('#'):
            header_match = HEADER_PATTERN.match(line)
            if header_match:
                _, line = header_match.groups()
                line_meta["type"] = "header"
        elif line.startswith('-'):
            bullet_match = BULLET_PATTERN.match(line)
            if bullet_match:
                line = bullet_match.group(1)
                line_meta["type"] = "bullet"
                line_meta["indent"] = 1
        elif line and line[0].isdigit():
            num_match = NUMBERED_PATTERN.match(line)
            if num_match:
                line = num_match.group(1)
                line_meta["type"] = "numbered"
                line_meta["indent"] = 1
        elif line.startswith('>'):
            blockquote_match = BLOCKQUOTE_PATTERN.match(line)
            if blockquote_match:
                line = blockquote_match.group(1)
                line_meta["type"] = "blockquote"
                line_meta["indent"] = 1

        # Replace smart punctuation
        for smart, normal in SMART_QUOTES.items():
            if smart in line:
                line = line.replace(smart, normal)

        # Remove markdown emphasis markers
        line = EMPHASIS_BOLD.sub("", line)
        line = EMPHASIS_ITALIC.sub("", line)

        # (Optional) remove or replace special characters
        line = re.sub(r'[^a-zA-Z0-9\s.,;:?!\'"-]', '', line)

        # 1) Split out any word > 7 chars as its own segment
        sub_lines = []
        if line:
            words = line.split()
            current_segment_words = []
            for w in words:
                if len(w) > 7:
                    if current_segment_words:
                        sub_lines.append(" ".join(current_segment_words))
                        current_segment_words = []
                    sub_lines.append(w)  # The big word stands alone
                else:
                    current_segment_words.append(w)
            if current_segment_words:
                sub_lines.append(" ".join(current_segment_words))
        else:
            sub_lines = [""]

        # 2) Wrap each final segment at 75 characters
        for segment in sub_lines:
            if len(segment) > 30:
                # Perform word-wrapping at 75 chars
                words_for_wrap = segment.split()
                current_line = []
                current_length = 0

                for word in words_for_wrap:
                    if current_line and (current_length + len(word) + 1 > 75):
                        results.append({"line": " ".join(current_line), "metadata": line_meta.copy()})
                        current_line = [word]
                        current_length = len(word)
                    else:
                        if not current_line:
                            current_line = [word]
                            current_length = len(word)
                        else:
                            current_line.append(word)
                            current_length += len(word) + 1

                if current_line:
                    results.append({"line": " ".join(current_line), "metadata": line_meta.copy()})
            else:
                results.append({"line": segment, "metadata": line_meta.copy()})

    processed_lines = [item["line"] for item in results]
    metadata = [item["metadata"] for item in results]
    return processed_lines, metadata

def metadata_to_style(metadata_list, style_id, lines):
    single_stroke_color = 'black'
    n = len(metadata_list)

    lines = lines.split("\n")
    biases = .2*np.flip(np.cumsum([len(i) == 0 for i in lines]), 0)
    # Using constant bias value, similar to test_syn.py
    biases = [0.75 for _ in range(n)]
    styles = [style_id for i in lines]

    pattern = [1, 1, 1, 1]
    stroke_widths = (pattern * ((n // len(pattern)) + 1))[:n]
    stroke_colors = [single_stroke_color] * n
    
    return styles, biases, stroke_colors, stroke_widths

def split_stroke_by_eos(stroke_coords):
    segments = []
    current_segment = []
    for pt in stroke_coords:
        current_segment.append(pt)
        if pt[2] == 1:
            segments.append(current_segment)
            current_segment = []
    if current_segment:
        current_segment[-1][2] = 1
        segments.append(current_segment)
    return segments

def optimize_strokes(strokes, precision=4):
    processed_strokes = []
    for stroke in strokes:
        processed_stroke = []
        for point in stroke:
            processed_point = [round(float(point[0]), precision), round(float(point[1]), precision)]
            processed_stroke.append(processed_point)
        processed_strokes.append(processed_stroke)
    return processed_strokes

def process_stroke(item, stroke, initial_coord, screen_width=800, screen_height=600):
    """
    Converts a single sub-segment (line item) into coordinate strokes
    at a given 'initial_coord', scaled for the given screen dimensions.
    """
    line = item["line"]
    meta = item["metadata"]
    sw = item["stroke_width"]
    sc = item["stroke_color"]

    if not line:
        return {
            "index": item["index"],
            "line": line,
            "strokes": [],
            "stroke_width": sw,
            "stroke_color": sc,
            "metadata": meta,
        }

    # Calculate responsive scaling factor based on screen width
    # Base scaling assumes 800px width, adjust proportionally
    base_width = 800
    scale_factor = (screen_width / base_width) * 1.5
    
    # Scale up based on screen size
    stroke[:, :2] *= scale_factor

    # Convert offsets -> absolute coords
    coords = drawing.offsets_to_coords(stroke)
    coords = drawing.denoise(coords)
    coords[:, :2] = drawing.align(coords[:, :2])
    coords[:, 1] *= -1

    # Place chunk around initial_coord
    coords[:, :2] = coords[:, :2] - coords[:, :2].min() + initial_coord

    # Responsive indentation based on screen width
    indent = meta.get("indent", 0)
    indent_size = max(20, screen_width * 0.025)  # 2.5% of screen width, minimum 20px
    coords[:, 0] += indent * indent_size + 20

    stroke_coords = coords.tolist()
    stroke_segments = split_stroke_by_eos(stroke_coords)
    stroke_segments = optimize_strokes(stroke_segments)

    return {
        "index": item["index"],
        "line": line,
        "strokes": stroke_segments,
        "stroke_width": sw,
        "stroke_color": sc,
        "metadata": meta
    }

def combine_segments_as_one_line(list_of_segments, screen_width=800, spacing=None):
    """
    Takes a list of sub-segment stroke data and places each 
    sub-segment horizontally after the previous one with responsive spacing.
    """
    # Calculate responsive spacing based on screen width
    if spacing is None:
        spacing = max(10, screen_width * 0.025)  # 2.5% of screen width, minimum 10px
    
    x_offset = 0.0
    combined_strokes = []

    for seg in list_of_segments:
        # seg is a list of strokes (each stroke is list of [x,y])
        min_x = float('inf')
        max_x = float('-inf')
        for stroke in seg:
            for (x, y) in stroke:
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x

        width_of_segment = max_x - min_x
        x_shift = x_offset - min_x

        # Shift all strokes in this segment
        shifted_segment = []
        for stroke in seg:
            shifted_stroke = []
            for (x, y) in stroke:
                shifted_stroke.append([x + x_shift, y])
            shifted_segment.append(shifted_stroke)

        # Add to combined list
        combined_strokes.extend(shifted_segment)

        # Advance x_offset for the next segment
        x_offset += width_of_segment + spacing

    return combined_strokes

def build_input_tensors(item, text_len):
    """
    Build the input tensors for the model, following test_syn.py approach
    """
    style_id = item["style"]
    style_strokes = np.load(f"styles/style-{style_id}-strokes.npy", allow_pickle=True)
    style_chars = np.load(f"styles/style-{style_id}-chars.npy", allow_pickle=True).tobytes().decode('ascii')
    text = item["line"]
    
    full_char_seq = drawing.encode_ascii(style_chars + ' ' + text)
    
    # Create arrays with proper shapes
    x_prime = np.zeros((1, 1200, 3), dtype=np.float32)
    x_prime_len = np.zeros((1,), dtype=np.int32)
    chars = np.zeros((1, 120), dtype=np.int32)
    chars_len = np.zeros((1,), dtype=np.int32)
    
    # Fill arrays with data
    chars[0, :len(full_char_seq)] = full_char_seq
    chars_len[0] = len(full_char_seq)
    
    x_prime[0, :len(style_strokes), :] = style_strokes
    x_prime_len[0] = len(style_strokes)
    
    max_tsteps = 40 * max(1, text_len)
    
    return x_prime, x_prime_len, chars, chars_len, max_tsteps

def render_strokes_to_image(strokes, out_path, dpi=150):
    """
    Renders a list of stroke sequences to a PNG file.
    """
    # 1) Compute global bounds so we can crop tightly
    all_x = [pt[0] for stroke in strokes for pt in stroke]
    all_y = [pt[1] for stroke in strokes for pt in stroke]
    if not all_x or not all_y:
        # Create empty image
        fig, ax = plt.subplots(figsize=(1, 1), dpi=dpi)
        ax.axis('off')
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return
        
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    width, height = max_x - min_x, max_y - min_y

    # 2) Create a figure sized roughly to the stroke extents
    fig, ax = plt.subplots(figsize=((width + 20) / 100, (height + 20) / 100), dpi=dpi)
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        # normalize to (0,0) origin
        xs = [x - min_x + 10 for x, y in stroke]
        ys = [y - min_y + 10 for x, y in stroke]
        ax.plot(xs, ys, color='black', linewidth=1)
    
    # invert y-axis
    ax.invert_yaxis()

    # 3) Tidy up axes
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout(pad=0)

    # 4) Save and close
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# Process a single item with proper thread safety
def process_single_item(item, ref_strokes=None, screen_width=800, screen_height=600):
    """
    Process a single line item into handwriting strokes.
    Thread-safe implementation using the global session.
    """
    text_len = max(1, len(item["line"]))
    
    # Build input tensors
    x_prime, x_prime_len, chars, chars_len, max_tsteps = build_input_tensors(item, text_len)
    
    # Use provided reference style strokes if available and not empty
    if ref_strokes is not None and len(ref_strokes) > 0:
        try:
            # Convert the list of objects to a list of [x, y, eos] arrays
            ref_strokes_array = np.array([[s['x'], s['y'], s['eos']] for s in ref_strokes], dtype=np.float32)
            
            # Normalize x and y coordinates to [-1, 1] range
            x_coords = ref_strokes_array[:, 0]
            y_coords = ref_strokes_array[:, 1]
            
            # Get min and max for x and y
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            # Calculate scale factors (using max range to maintain aspect ratio)
            x_range = x_max - x_min
            y_range = y_max - y_min
            max_range = max(x_range, y_range)
            
            if max_range > 0:
                # Normalize to [-1, 1] range
                ref_strokes_array[:, 0] = 2 * (x_coords - x_min) / max_range - 1
                ref_strokes_array[:, 1] = 2 * (y_coords - y_min) / max_range - 1
            
            x_prime = ref_strokes_array.reshape(1, -1, 3)
            x_prime_len = np.array([len(ref_strokes_array)], dtype=np.int32)
            logging.info(f"Using normalized reference strokes with shape {x_prime.shape}")
        except Exception as e:
            logging.error(f"Error with reference strokes: {e}")
            # Continue with default style
    
    # Use the thread lock to ensure only one thread uses the TF session at a time
    with tf_lock:
        # Sample handwriting sequence
        [sampled] = style_model.session.run(
            [style_model.sampled_sequence],
            feed_dict={
                style_model.prime: True,
                style_model.x_prime: x_prime,
                style_model.x_prime_len: x_prime_len,
                style_model.ref_x: x_prime,
                style_model.ref_x_len: x_prime_len,
                style_model.num_samples: 1,
                style_model.sample_tsteps: max_tsteps,
                style_model.c: chars,
                style_model.c_len: chars_len,
                style_model.bias: [item["bias"]]
            }
        )
    
    # Strip padding rows (any all-zero rows)
    seq = sampled[0][~np.all(sampled[0] == 0.0, axis=1)]
    
    # Process the stroke sequence with responsive dimensions
    base_initial_coord = np.array([0, -30])
    initial_coord = base_initial_coord.copy()
    processed = process_stroke(item, seq, initial_coord, screen_width, screen_height)
    
    logging.info(f"Processed line {item['index']}")
    return processed

@app.get("/health")
def health():
    region = os.environ.get('CLOUD_RUN_REGION', 'unknown')
    vm_name = os.environ.get('HOSTNAME', 'unknown')
    return {"message": "OK", "region": region, "instance": os.environ.get('K_REVISION', 'unknown'), "vm": vm_name}

@app.get("/hello")
def hello():
    return {"message": "Hello, World!"}

def save_request_data(request: MarkdownRequest, output_dir: str = "saved_requests"):
    """
    Save the incoming request data to a JSON file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"request_{timestamp}_{uuid.uuid4().hex[:8]}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Convert request to dict and save
    request_data = {
        "markdown": request.markdown,
        "style_id": request.style_id,
        "ref_strokes": request.ref_strokes,
        "screen_width": request.screen_width,
        "screen_height": request.screen_height,
        "timestamp": timestamp
    }
    
    with open(filepath, 'w') as f:
        json.dump(request_data, f, indent=2)
    
    return filepath

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
        - If reference strokes are provided, a quality check is performed
        - If the quality check fails, it falls back to default style generation
        - The endpoint maintains markdown formatting like headers, lists, and indentation
    """
    # Save the incoming request data
    try:
        saved_file = save_request_data(request)
        logging.info(f"Saved request data to {saved_file}")
    except Exception as e:
        logging.error(f"Failed to save request data: {e}")
    
    # 1) Parse markdown → lines + metadata
    try:
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

    # Determine optimal number of threads based on CPU count
    cpu_count = os.cpu_count() or 4
    # Use 2x CPU count for I/O bound tasks, but cap to avoid too much overhead
    max_workers = min(cpu_count * 2, 16)
    
    logging.info(f"Using {max_workers} worker threads with {cpu_count} CPU cores")

    # Check if reference style strokes were provided
    ref_strokes = request.ref_strokes
    screen_width = request.screen_width
    screen_height = request.screen_height
    
    # Calculate responsive line spacing based on screen height
    base_height = 600
    line_spacing = max(30, (screen_height / base_height) * 50)
    
    # Process items in parallel using thread pool
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all items to the executor with screen dimensions
            future_to_item = {executor.submit(process_single_item, item, ref_strokes, screen_width, screen_height): item 
                            for item in indexed_items}
            
            # Collect results as they complete
            processed_lines = []
            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    processed_lines.append(result)
                    # Log progress every 10 items
                    if len(processed_lines) % 10 == 0:
                        logging.info(f"Processed {len(processed_lines)}/{len(indexed_items)} items")
                except Exception as e:
                    logging.error(f"Processing failed for item {item['index']}: {e}")
                    # Create a minimal placeholder for failed items
                    processed_lines.append({
                        "index": item["index"],
                        "line": item["line"],
                        "strokes": [],
                        "stroke_width": item["stroke_width"],
                        "stroke_color": item["stroke_color"],
                        "metadata": item["metadata"]
                    })
    except Exception as e:
        # Fall back to sequential processing if parallel execution fails
        logging.error(f"Thread pool execution failed, falling back to sequential: {e}")
        processed_lines = []
        for item in indexed_items:
            try:
                processed = process_single_item(item, ref_strokes, screen_width, screen_height)
                processed_lines.append(processed)
            except Exception as item_e:
                logging.error(f"Sequential processing failed for item {item['index']}: {item_e}")
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
    for gid in sorted(grouped):
        group = grouped[gid]
        text = " ".join(filter(None, group["lines"]))
        combined = combine_segments_as_one_line(group["segments"], screen_width)

        # Responsive vertical offset by line number
        for stroke in combined:
            for pt in stroke:
                pt[1] += gid * line_spacing

        # Responsive indentation if needed
        indent = group["metadata"].get("indent", 0)
        if indent:
            indent_size = max(20, screen_width * 0.025)  # 2.5% of screen width, minimum 20px
            for stroke in combined:
                for pt in stroke:
                    pt[0] += indent * indent_size + 20

        merged_output.append({
            "line": text,
            "strokes": combined,
            "stroke_width": group["stroke_width"],
            "stroke_color": group["stroke_color"]
        })

    # If we have ref_strokes, check the quality of the generated handwriting
    if ref_strokes:
        try:
            # Create a temporary file for the image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save all strokes to the temporary image
            all_strokes = [stroke for item in merged_output for stroke in item["strokes"]]
            save_strokes_to_image(all_strokes, temp_path)
            
            # Check handwriting quality
            quality_check = check_handwriting_quality(temp_path)
            
            # Clean up the temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logging.error(f"Error cleaning up temporary file: {e}")
            
            # If quality check passes, return the current strokes
            if quality_check:
                logging.info("Handwriting quality check passed with reference strokes")
                return {"strokes": merged_output}
            
            # If quality check fails, regenerate with default style
            logging.info("Handwriting quality check failed, regenerating with default style")
            return convert_markdown(MarkdownRequest(
                markdown=request.markdown,
                style_id=request.style_id,
                ref_strokes=None,
                screen_width=request.screen_width,
                screen_height=request.screen_height
            ))
            
        except Exception as e:
            logging.error(f"Error during handwriting quality check: {e}")
            # If there's an error in quality check, fall back to default style
            return convert_markdown(MarkdownRequest(
                markdown=request.markdown,
                style_id=request.style_id,
                ref_strokes=None,
                screen_width=request.screen_width,
                screen_height=request.screen_height
            ))
    
    # If no ref_strokes provided, return the current strokes
    return {"strokes": merged_output}

def save_strokes_to_image(strokes, output_path):
    """
    Saves the strokes to an image file.
    """
    # Create figure with white background
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Plot each stroke
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        xs = [pt[0] for pt in stroke]
        ys = [pt[1] for pt in stroke]
        ax.plot(xs, ys, color='black', linewidth=1)
    
    # Remove axes and set equal aspect
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)

def check_handwriting_quality(image_path):
    """
    Calls the handwriting quality check API.
    """
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{GEMINI_SERVICE_URL}/check_handwriting", files=files, timeout=5)
            response.raise_for_status()
            result = response.json()
            return result.get('ok', False) and result.get('result', False)
    except requests.exceptions.RequestException as e:
        logging.warning(f"Handwriting quality service unavailable: {e}")
        # If service is unavailable, assume quality is good
        return True
    except Exception as e:
        logging.error(f"Error checking handwriting quality: {e}")
        # If there's an error, assume quality is good
        return True

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