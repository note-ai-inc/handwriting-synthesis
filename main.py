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
# Use non-interactive backend for better performance
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading
import concurrent.futures
from functools import partial

# Configure TensorFlow for better performance
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
    ref_style_strokes: Optional[list] = None  # Optional reference style strokes

# Set up the model
DATA_DIR = "data/processed"
reader = StyleDataReader(DATA_DIR)
style_model = StyleSynthesisModel(
    reader=reader,
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

def process_stroke(item, stroke, initial_coord):
    """
    Converts a single sub-segment (line item) into coordinate strokes
    at a given 'initial_coord'.
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

    # Scale up slightly
    stroke[:, :2] *= 1.5

    # Convert offsets -> absolute coords
    coords = drawing.offsets_to_coords(stroke)
    coords = drawing.denoise(coords)
    coords[:, :2] = drawing.align(coords[:, :2])
    coords[:, 1] *= -1

    # Place chunk around initial_coord
    coords[:, :2] = coords[:, :2] - coords[:, :2].min() + initial_coord

    indent = meta.get("indent", 0)
    coords[:, 0] += indent * 50 + 20

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

def combine_segments_as_one_line(list_of_segments, spacing=20):
    """
    Takes a list of sub-segment stroke data and places each 
    sub-segment horizontally after the previous one.
    """
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
    style_strokes = np.load(f"styles/style-{style_id}-strokes.npy")
    style_chars = np.load(f"styles/style-{style_id}-chars.npy").tobytes().decode('ascii')
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
def process_single_item(item, ref_style_strokes=None):
    """
    Process a single line item into handwriting strokes.
    Thread-safe implementation using the global session.
    """
    text_len = max(1, len(item["line"]))
    
    # Build input tensors
    x_prime, x_prime_len, chars, chars_len, max_tsteps = build_input_tensors(item, text_len)
    
    # Use provided reference style strokes if available
    if ref_style_strokes is not None:
        try:
            ref_strokes = np.array(ref_style_strokes, dtype=np.float32)
            x_prime = ref_strokes.reshape(1, -1, 3)
            x_prime_len = np.array([len(ref_strokes)], dtype=np.int32)
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
    
    # Process the stroke sequence
    base_initial_coord = np.array([0, -30])
    initial_coord = base_initial_coord.copy()
    processed = process_stroke(item, seq, initial_coord)
    
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
    """
    1. Parse the incoming Markdown into lines & metadata.
    2. Compute style/bias/etc. for each line.
    3. Process lines in parallel using thread pool.
    4. Merge all processed lines back into the full-page strokes JSON.
    """
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

    # 3) Thread-based parallel processing for high performance
    # Using thread pool is more reliable than multiprocessing with TensorFlow
    base_initial_coord = np.array([0, -30])
    
    # Determine optimal number of threads based on CPU count
    cpu_count = os.cpu_count() or 4
    # Use 2x CPU count for I/O bound tasks, but cap to avoid too much overhead
    max_workers = min(cpu_count * 2, 16)
    
    logging.info(f"Using {max_workers} worker threads with {cpu_count} CPU cores")
    
    # Check if reference style strokes were provided
    ref_strokes = request.ref_style_strokes
    
    # Process items in parallel using thread pool
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all items to the executor
            future_to_item = {executor.submit(process_single_item, item, ref_strokes): item 
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
                processed = process_single_item(item, ref_strokes)
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

    # 4) Merge per-line results into full-page output
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
        combined = combine_segments_as_one_line(group["segments"], spacing=20)

        # Vertical offset by line number
        for stroke in combined:
            for pt in stroke:
                pt[1] += gid * 50

        # Indent if needed
        indent = group["metadata"].get("indent", 0)
        if indent:
            for stroke in combined:
                for pt in stroke:
                    pt[0] += indent * 50 + 20

        merged_output.append({
            "line": text,
            "strokes": combined,
            "stroke_width": group["stroke_width"],
            "stroke_color": group["stroke_color"]
        })

    return {"strokes": merged_output}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)