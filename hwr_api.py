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
checkpoint_path = "checkpoints/model-10350"
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
BULLET_PATTERN = re.compile(r"^(\s*)-\s+(.*)")  # Capture leading spaces for nesting
NUMBERED_PATTERN = re.compile(r"^(\s*)\d+\.\s+(.*)")  # Capture leading spaces for nesting
BLOCKQUOTE_PATTERN = re.compile(r"^(\s*)>\s+(.*)")  # Capture leading spaces for nesting
EMPHASIS_BOLD = re.compile(r"(\*\*|__)")
EMPHASIS_ITALIC = re.compile(r"(\*|_)")

def parse_markdown(markdown_text):
    """
    Parses a markdown string and returns:
      - a list of text lines (with markdown markers removed and normalized)
      - a list of metadata dictionaries corresponding to each line with enhanced formatting info.
    """
    SMART_QUOTES = {
        """: '"', """: '"', "'": "'",
        "'": "'", "–": "-", "—": "-"
    }
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.*)")
    BULLET_PATTERN = re.compile(r"^(\s*)-\s+(.*)")  # Capture leading spaces for nesting
    NUMBERED_PATTERN = re.compile(r"^(\s*)\d+\.\s+(.*)")  # Capture leading spaces for nesting
    BLOCKQUOTE_PATTERN = re.compile(r"^(\s*)>\s+(.*)")  # Capture leading spaces for nesting
    EMPHASIS_BOLD = re.compile(r"(\*\*|__)")
    EMPHASIS_ITALIC = re.compile(r"(\*|_)")

    results = []
    line_meta_base = {
        "type": "paragraph", 
        "indent": 0, 
        "header_level": 0,
        "is_list_item": False,
        "list_type": None,
        "nesting_level": 0,
        "spacing_before": 0.5,  # Multiplier for spacing before this element
        "spacing_after": 0.5,   # Multiplier for spacing after this element
        "font_scale": 1.0       # Scale factor for text size
    }

    for line_idx, raw_line in enumerate(markdown_text.splitlines()):
        line = raw_line.strip()
        line_meta = line_meta_base.copy()

        # Assign each original raw line an ID
        line_meta["group_id"] = line_idx

        # Detect markdown constructs with enhanced metadata
        if line.startswith('#'):
            header_match = HEADER_PATTERN.match(line)
            if header_match:
                header_markers, line = header_match.groups()
                header_level = len(header_markers)
                line_meta["type"] = "header"
                line_meta["header_level"] = header_level
                
                # Header spacing and scaling based on level
                if header_level == 1:  # H1
                    line_meta["spacing_before"] = 0.6  # Reduced from 2.5
                    line_meta["spacing_after"] = 0.6   # Reduced from 1.8
                    line_meta["font_scale"] = 1.8
                elif header_level == 2:  # H2
                    line_meta["spacing_before"] = 0.5  # Reduced from 2.0
                    line_meta["spacing_after"] = 0.5   # Reduced from 1.5
                    line_meta["font_scale"] = 1.5
                elif header_level == 3:  # H3
                    line_meta["spacing_before"] = 0.5  # Reduced from 1.8
                    line_meta["spacing_after"] = 0.5   # Reduced from 1.3
                    line_meta["font_scale"] = 1.3
                elif header_level == 4:  # H4
                    line_meta["spacing_before"] = 0.5  # Reduced from 1.5
                    line_meta["spacing_after"] = 0.5   # Reduced from 1.2
                    line_meta["font_scale"] = 1.2
                elif header_level == 5:  # H5
                    line_meta["spacing_before"] = 0.5  # Reduced from 1.3
                    line_meta["spacing_after"] = 0.5   # Reduced from 1.1
                    line_meta["font_scale"] = 1.1
                else:  # H6
                    line_meta["spacing_before"] = 0.5  # Reduced from 1.2
                    line_meta["spacing_after"] = 0.5   # Same
                    line_meta["font_scale"] = 1.05
                    
        elif line.startswith('-') or (raw_line.startswith(' ') and '-' in raw_line):
            bullet_match = BULLET_PATTERN.match(raw_line)  # Use raw_line to capture indentation
            if bullet_match:
                leading_spaces, line = bullet_match.groups()
                nesting_level = len(leading_spaces) // 2  # 2 spaces per nesting level
                line_meta["type"] = "bullet"
                line_meta["is_list_item"] = True
                line_meta["list_type"] = "bullet"
                line_meta["nesting_level"] = nesting_level
                line_meta["indent"] = 1 + nesting_level
                line_meta["spacing_before"] = 0.8 if nesting_level == 0 else 0.6
                line_meta["spacing_after"] = 0.8 if nesting_level == 0 else 0.6
                
        elif line and (line[0].isdigit() or (raw_line.startswith(' ') and any(c.isdigit() for c in raw_line))):
            num_match = NUMBERED_PATTERN.match(raw_line)  # Use raw_line to capture indentation
            if num_match:
                leading_spaces, line = num_match.groups()
                nesting_level = len(leading_spaces) // 2  # 2 spaces per nesting level
                line_meta["type"] = "numbered"
                line_meta["is_list_item"] = True
                line_meta["list_type"] = "numbered"
                line_meta["nesting_level"] = nesting_level
                line_meta["indent"] = 1 + nesting_level
                line_meta["spacing_before"] = 0.8 if nesting_level == 0 else 0.6
                line_meta["spacing_after"] = 0.8 if nesting_level == 0 else 0.6
                
        elif line.startswith('>') or (raw_line.startswith(' ') and '>' in raw_line):
            blockquote_match = BLOCKQUOTE_PATTERN.match(raw_line)  # Use raw_line to capture indentation
            if blockquote_match:
                leading_spaces, line = blockquote_match.groups()
                nesting_level = len(leading_spaces) // 2
                line_meta["type"] = "blockquote"
                line_meta["nesting_level"] = nesting_level
                line_meta["indent"] = 1 + nesting_level
                line_meta["spacing_before"] = 1.2
                line_meta["spacing_after"] = 1.2
                line_meta["font_scale"] = 0.95  # Slightly smaller for quotes
        
        # Handle empty lines (paragraph breaks)
        elif not line.strip():
            line_meta["type"] = "empty"
            line_meta["spacing_before"] = 0.5
            line_meta["spacing_after"] = 0.5

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

        # 2) Wrap each final segment at responsive character limit based on screen width
        for segment in sub_lines:
            # Responsive character limit (will be passed from screen width)
            char_limit = 75  # Default, will be adjusted in calling function
            
            if len(segment) > 30:
                # Perform word-wrapping at char_limit
                words_for_wrap = segment.split()
                current_line = []
                current_length = 0

                for word in words_for_wrap:
                    if current_line and (current_length + len(word) + 1 > char_limit):
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

def ensure_bounds(strokes, screen_width, screen_height, margin=10):
    """
    Ensure all stroke coordinates stay within the canvas bounds with a margin.
    This function scales and translates coordinates to fit within the canvas.
    """
    if not strokes:
        return strokes
    
    # Find the bounding box of all strokes
    all_x = []
    all_y = []
    for stroke in strokes:
        for pt in stroke:
            all_x.append(pt[0])
            all_y.append(pt[1])
    
    if not all_x or not all_y:
        return strokes
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # Calculate current dimensions
    current_width = max_x - min_x
    current_height = max_y - min_y
    
    # Calculate available space (with margins)
    available_width = screen_width - 2 * margin
    available_height = screen_height - 2 * margin
    
    # Calculate scale factors to fit within bounds
    scale_x = available_width / current_width if current_width > 0 else 1.0
    scale_y = available_height / current_height if current_height > 0 else 1.0
    
    # Use the smaller scale factor to maintain aspect ratio
    scale = min(scale_x, scale_y, 1.0)  # Don't scale up, only down if needed
    
    # Apply scaling and translation to fit within bounds
    for stroke in strokes:
        for pt in stroke:
            # Scale coordinates
            pt[0] = (pt[0] - min_x) * scale + margin
            pt[1] = (pt[1] - min_y) * scale + margin
    
    return strokes

def process_stroke(item, stroke, initial_coord, screen_width=800, screen_height=600):
    """
    Converts a single sub-segment (line item) into coordinate strokes
    at a given 'initial_coord', scaled for the given screen dimensions.
    Now supports font scaling for headers and improved indentation.
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

    # Calculate a responsive scaling factor for handwriting size:
    # - (screen_width / base_width): scales proportionally to the screen width (base is 800px)
    # - * 1.1: increases the base size by 10% to make the handwriting larger and more readable.
    # - min(..., 1.4): caps the scaling factor at 1.4 to prevent handwriting from becoming excessively large.
    base_width = 800
    base_scale_factor = min((screen_width / base_width) * 1.1, 1.4)
    
    # Apply font scaling for headers and other elements
    font_scale = meta.get("font_scale", 1.0)
    scale_factor = base_scale_factor * font_scale
    
    # Scale up based on screen size and font requirements
    stroke[:, :2] *= scale_factor

    # Convert offsets -> absolute coords
    coords = drawing.offsets_to_coords(stroke)
    coords = drawing.denoise(coords)
    coords[:, :2] = drawing.align(coords[:, :2])
    coords[:, 1] *= -1

    # Place chunk around initial_coord with bounds consideration
    # The .min(axis=0) ensures we normalize x and y independently,
    # preserving the aspect ratio of the generated strokes.
    coords[:, :2] = coords[:, :2] - coords[:, :2].min(axis=0) + initial_coord

    stroke_coords = coords.tolist()
    stroke_segments = split_stroke_by_eos(stroke_coords)
    stroke_segments = optimize_strokes(stroke_segments)

    # Note: Removed ensure_bounds here to preserve line spacing that will be added later

    return {
        "index": item["index"],
        "line": line,
        "strokes": stroke_segments,
        "stroke_width": sw,
        "stroke_color": sc,
        "metadata": meta
    }

def combine_segments_as_one_line(list_of_segments, screen_width=800, screen_height=600, spacing=None):
    """
    Takes a list of sub-segment stroke data and places each 
    sub-segment horizontally with automatic wrapping when exceeding screen width.
    Maintains rightward shift and proper bullet spacing.
    """
    if not list_of_segments:
        return []
    
    # Calculate responsive layout parameters
    margin = max(15, screen_width * 0.05)  # 5% margin (minimum 15px)
    fixed_right_shift = screen_width * 0.3  # Matches ensure_bounds_preserve_spacing
    available_width = screen_width - 2 * margin - fixed_right_shift
    line_height = max(40, screen_height * 0.08)  # Responsive line height
    
    x_offset = margin + fixed_right_shift  # Start position with rightward shift
    y_offset = 0
    combined_strokes = []

    for seg in list_of_segments:
        if not seg or not any(seg):
            continue
            
        # Calculate segment dimensions
        all_x = [pt[0] for stroke in seg for pt in stroke]
        all_y = [pt[1] for stroke in seg for pt in stroke]
        width = max(all_x) - min(all_x) if all_x else 0
        height = max(all_y) - min(all_y) if all_y else 0
        
        # Wrap to new line if segment exceeds available width
        if x_offset + width > available_width and x_offset > margin + fixed_right_shift:
            x_offset = margin + fixed_right_shift
            y_offset += line_height
        
        # Calculate required shifts
        x_shift = x_offset - (min(all_x) if all_x else 0)
        y_shift = y_offset - (min(all_y) if all_y else 0)
        
        # Apply shifts to all points in segment
        shifted_segment = []
        for stroke in seg:
            shifted_stroke = [[pt[0] + x_shift, pt[1] + y_shift] for pt in stroke]
            shifted_segment.append(shifted_stroke)
        
        combined_strokes.extend(shifted_segment)
        
        # Update x_offset with dynamic spacing
        x_offset += width + (spacing if spacing is not None else max(8, width * 0.2))

    return combined_strokes

def calculate_indentation(metadata, screen_width):
    """
    Calculate the indentation offset for a given line based on its metadata.
    """
    indent = metadata.get("indent", 0)
    nesting_level = metadata.get("nesting_level", 0)
    element_type = metadata.get("type", "paragraph")
    
    # Calculate base indent size responsive to screen width
    base_indent_size = max(20, screen_width * 0.025)  # 2.5% of screen width, minimum 20px
    
    indent_offset = 0
    # Different indentation strategies for different element types
    if element_type == "header":
        indent_offset = base_indent_size * 0.2
    elif element_type in ["bullet", "numbered"]:
        list_base_indent = base_indent_size * 2.5
        nesting_indent = nesting_level * base_indent_size * 1.5
        indent_offset = list_base_indent + nesting_indent
        
        if element_type == "bullet":
            indent_offset += base_indent_size * 0.3
        else:
            indent_offset += base_indent_size * 0.5
            
    elif element_type == "blockquote":
        quote_base_indent = base_indent_size * 2.0
        nesting_indent = nesting_level * base_indent_size * 0.6
        indent_offset = quote_base_indent + nesting_indent
    else:
        indent_offset = base_indent_size * 5.0
    
    # Ensure indentation doesn't exceed reasonable bounds
    max_indent_offset = screen_width * 0.40
    indent_offset = min(indent_offset, max_indent_offset)
    
    return indent_offset

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
    # Use 2x CPU count for I/O bound tasks, but cap to avoid too much overhead
    max_workers = min(cpu_count * 2, 16)
    
    logging.info(f"Using {max_workers} worker threads with {cpu_count} CPU cores")
    
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

def ensure_bounds_preserve_spacing(merged_output, screen_width, screen_height, margin=10):
    """
    Ensure all stroke coordinates stay within canvas bounds while preserving 
    the relative spacing between lines and elements.
    Enforces a minimum scale to make strokes larger.
    """
    if not merged_output:
        return merged_output
    
    # Find global bounds across all stroke groups
    all_x = []
    all_y = []
    for group in merged_output:
        for stroke in group["strokes"]:
            for pt in stroke:
                all_x.append(pt[0])
                all_y.append(pt[1])
    
    if not all_x or not all_y:
        return merged_output
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # Calculate current dimensions
    current_width = max_x - min_x
    current_height = max_y - min_y
    
    # Calculate available space (with margins)
    available_width = screen_width - 2 * margin
    available_height = screen_height - 2 * margin
    
    # Calculate scale factors to fit within bounds
    scale_x = available_width / current_width if current_width > 0 else 1.0
    scale_y = available_height / current_height if current_height > 0 else 1.0
    
    # Use the smaller scale factor to maintain aspect ratio
    scale = min(scale_x, scale_y)
    
    # Enforce a minimum scale to make strokes larger (e.g., 1.5x)
    min_scale = 0.8
    scale = max(scale, min_scale)  # Ensure strokes are never smaller than min_scale
    
    # Apply scaling to make strokes larger while preserving relative positions
    for group in merged_output:
        for stroke in group["strokes"]:
            for pt in stroke:
                # Scale coordinates around the origin, preserving relative positions
                pt[0] = pt[0] * scale
                pt[1] = pt[1] * scale
    
    # Recalculate bounds after scaling
    all_x = []
    all_y = []
    for group in merged_output:
        for stroke in group["strokes"]:
            for pt in stroke:
                all_x.append(pt[0])
                all_y.append(pt[1])
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # Apply translation to fit within bounds
    shift_x = 0
    shift_y = 0
    
    if min_x < margin:
        shift_x = margin - min_x
    elif max_x > screen_width - margin:
        shift_x = (screen_width - margin) - max_x
    
    if min_y < margin:
        shift_y = margin - min_y
    elif max_y > screen_height - margin:
        shift_y = (screen_height - margin) - max_y
    
    # Apply a fixed rightward shift (e.g., 100 pixels for more movement to the right)
    fixed_right_shift = screen_width * 0.125  # Make shift responsive
    shift_x += fixed_right_shift
    
    # Apply translation if needed
    if shift_x != 0 or shift_y != 0:
        for group in merged_output:
            for stroke in group["strokes"]:
                for pt in stroke:
                    pt[0] += shift_x
                    pt[1] += shift_y
    
    return merged_output

def calculate_dynamic_spacing(metadata, screen_width, screen_height, base_line_spacing):
    """
    Calculate dynamic spacing for a line based on its metadata and screen dimensions.
    Returns the vertical offset to add for this line.
    Now uses increased spacing for bullet points only.
    """
    element_type = metadata.get("type", "paragraph")
    spacing_before = metadata.get("spacing_before", 1.0)
    header_level = metadata.get("header_level", 0)
    
    # Base spacing calculation
    spacing = base_line_spacing * spacing_before * 0.2
    
    # Adjusted spacing multipliers
    if element_type == "header":
        if header_level == 1:
            spacing *= 1.1
        elif header_level == 2:
            spacing *= 1.05
        elif header_level == 3:
            spacing *= 1.0
        else:
            spacing *= 0.95
            
    elif element_type in ["bullet", "numbered"]:
        nesting_level = metadata.get("nesting_level", 0)
        if nesting_level == 0:
            spacing *= 1.0  # Increased for top-level lists
        else:
            spacing *= 0.8  # Increased for nested lists
            
    elif element_type == "blockquote":
        spacing *= 0.9
        
    elif element_type == "empty":
        spacing *= 0.2
    
    # Bounds
    min_spacing = base_line_spacing * 0.1
    max_spacing = base_line_spacing * 1.5
    
    return max(min_spacing, min(spacing, max_spacing))

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