import os
import re
import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from rnn import rnn
import drawing

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
    style_id: Optional[int] = 8  # Default style id is 8 if not provided



def parse_markdown(markdown_text):
    """
    Parses a markdown string and returns:
      - a list of text lines (with markdown markers removed and normalized)
      - a list of metadata dictionaries corresponding to each line.
    """
    SMART_QUOTES = {
        "“": '"', "”": '"', "‘": "'",
        "’": "'", "–": "-", "—": "-"
    }
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.*)")
    BULLET_PATTERN = re.compile(r"^-\s+(.*)")
    NUMBERED_PATTERN = re.compile(r"^\d+\.\s+(.*)")
    BLOCKQUOTE_PATTERN = re.compile(r"^>\s+(.*)")
    EMPHASIS_BOLD = re.compile(r"(\*\*|__)")
    EMPHASIS_ITALIC = re.compile(r"(\*|_)")

    results = []
    line_meta_base = {"type": "paragraph", "indent": 0}

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        line_meta = line_meta_base.copy()

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
        
        # If the line is longer than 75 characters, split it into multiple lines
        if len(line) > 75:
            words = line.split()
            current_line = []
            current_length = 0
            
            for word in words:
                word_len = len(word)
                if current_line and (current_length + word_len + 1 > 75):
                    results.append({"line": " ".join(current_line), "metadata": line_meta.copy()})
                    current_line = [word]
                    current_length = word_len
                else:
                    current_line.append(word)
                    # Add 1 for the space, but only if it's not the first word
                    current_length += word_len + (1 if current_line else 0)
            
            if current_line:
                results.append({"line": " ".join(current_line), "metadata": line_meta.copy()})
        else:
            results.append({"line": line, "metadata": line_meta})
    
    processed_lines = [item["line"] for item in results]
    metadata = [item["metadata"] for item in results]
    return processed_lines, metadata


def metadata_to_style(metadata_list, style_id):
    single_bias = 0.90        # fixed bias for every line
    single_stroke_color = 'black'
    n = len(metadata_list)
    styles = [style_id] * n
    biases = [single_bias] * n
    # Repeat the pattern [2, 2, 2, 2] to match the number of lines
    pattern = [1, 1, 1, 1]
    stroke_widths = (pattern * ((n // len(pattern)) + 1))[:n]
    stroke_colors = [single_stroke_color] * n
    return styles, biases, stroke_colors, stroke_widths


def split_stroke_by_eos(stroke_coords):
    """
    Splits a list of stroke coordinates (each point is [x, y, eos])
    into separate segments. Each time a point with eos==1 is encountered,
    the current segment is ended and added to the list.
    If the final segment does not end with eos==1, the last point is forced to 1.
    """
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


class Hand:
    def __init__(self):
        # Use GPU if available; ensure your TensorFlow installation is GPU-enabled
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        # Initialize the handwriting synthesis RNN model.
        self.nn = rnn(
            log_dir='logs',
            checkpoint_dir='checkpoints',
            prediction_dir='predictions',
            learning_rates=[.0001, .00005, .00002],
            batch_sizes=[32, 64, 64],
            patiences=[1500, 1000, 500],
            beta1_decays=[.9, .9, .9],
            validation_batch_size=32,
            optimizer='rms',
            num_training_steps=100000,
            warm_start_init_step=17900,
            regularization_constant=0.0,
            keep_prob=1.0,
            enable_parameter_averaging=False,
            min_steps_to_checkpoint=2000,
            log_interval=20,
            logging_level=logging.CRITICAL,
            grad_clip=10,
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10
        )
        self.nn.restore()

    def _sample(self, lines, biases=None, styles=None):
        """
        Generates stroke instructions from text lines using the handwriting RNN.
        """
        num_samples = len(lines)
        max_tsteps = 40 * max(len(line) for line in lines)
        biases = biases if biases is not None else [0.5] * num_samples

        x_prime = np.zeros([num_samples, 1200, 3])
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120])
        chars_len = np.zeros([num_samples])

        if styles is not None:
            for i, (cs, style) in enumerate(zip(lines, styles)):
                x_p = np.load(f'styles/style-{style}-strokes.npy')
                c_p = np.load(f'styles/style-{style}-chars.npy').tostring().decode('utf-8')
                
                # Prime with the style strokes and append the target text
                c_p = str(c_p) + " " + cs
                c_p = drawing.encode_ascii(c_p)
                c_p = np.array(c_p)
                
                x_prime[i, :len(x_p), :] = x_p
                x_prime_len[i] = len(x_p)
                chars[i, :len(c_p)] = c_p
                chars_len[i] = len(c_p)
        else:
            for i, text_line in enumerate(lines):
                encoded = drawing.encode_ascii(text_line)
                chars[i, :len(encoded)] = encoded
                chars_len[i] = len(encoded)

        [samples] = self.nn.session.run(
            [self.nn.sampled_sequence],
            feed_dict={
                self.nn.prime: styles is not None,
                self.nn.x_prime: x_prime,
                self.nn.x_prime_len: x_prime_len,
                self.nn.num_samples: num_samples,
                self.nn.sample_tsteps: max_tsteps,
                self.nn.c: chars,
                self.nn.c_len: chars_len,
                self.nn.bias: biases
            }
        )
        # Remove trailing zeros from each sample
        samples = [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]
        return samples

def optimize_strokes(strokes, precision=4):
    """
    Reduces the precision of the stroke coordinates and removes the pressure value.
    Assumes each stroke is a list of coordinates [x, y, pressure], where pressure is the third value.
    
    :param strokes: List of strokes, where each stroke is a list of coordinates.
    :param precision: The number of decimal places to round the coordinates.
    :return: A list of processed strokes.
    """

    processed_strokes = []

    for stroke in strokes:
        # Process each stroke's coordinates
        processed_stroke = []
        
        for point in stroke:
            # Round the x, y coordinates to the given precision (remove pressure value)
            processed_point = [round(float(point[0]), precision), round(float(point[1]), precision)]
            processed_stroke.append(processed_point)
        
        processed_strokes.append(processed_stroke)

    return processed_strokes


def process_stroke(item, stroke, initial_coord):
    """
    Process a single stroke: scales, converts offsets to coordinates, denoises,
    aligns, adjusts for indent, and splits the stroke by end-of-sequence markers.
    The starting coordinate is computed per-line so that processing is independent.
    """
    line = item["line"]
    meta = item["metadata"]
    sw = item["stroke_width"]
    sc = item["stroke_color"]

    # If the line is empty, return an empty stroke
    if not line:
        return {
            "index": item["index"],
            "line": line,
            "strokes": [],
            "stroke_width": sw,
            "stroke_color": sc
        }

    # Scale the stroke offsets
    stroke[:, :2] *= 1.5

    # Convert offsets to (x,y,eos) coordinates
    coords = drawing.offsets_to_coords(stroke)
    coords = drawing.denoise(coords)
    coords[:, :2] = drawing.align(coords[:, :2])
    coords[:, 1] *= -1

    # Normalize coordinates relative to the minimum and adjust by the initial coordinate
    coords[:, :2] = coords[:, :2] - coords[:, :2].min() + initial_coord

    # Adjust horizontal position based on metadata indent and a fixed offset
    indent = meta.get("indent", 0)
    coords[:, 0] += indent * 50 + 20

    stroke_coords = coords.tolist()
    stroke_segments = split_stroke_by_eos(stroke_coords)

    # stroke_segments = optimize_strokes(stroke_segments)

    return {
        "index": item["index"],
        "line": line,
        "strokes": stroke_segments,
        "stroke_width": sw,
        "stroke_color": sc
    }

@app.get("/health")
def health():
    region = os.environ.get('CLOUD_RUN_REGION', 'unknown')
    return {
        "message": "OK",
        "region": region,
        "instance": os.environ.get('K_REVISION', 'unknown')
    }


@app.get("/hello")
def hello():
    return {"message": "Hello, World!"}


@app.post("/convert")
def convert_markdown(request: MarkdownRequest):
    """
    Accepts markdown text and a style_id, parses and synthesizes handwriting strokes,
    adjusts the (x,y) coordinates for rendering, and returns the strokes.
    """
    markdown_text = request.markdown
    try:
        lines, metadata = parse_markdown(markdown_text)
        # for i, (line, meta) in enumerate(zip(lines, metadata)):
        #     print(f"Line {i}: {line} with metadata: {meta}")

        # Use the provided style_id for all lines
        styles, biases, stroke_colors, stroke_widths = metadata_to_style(metadata, style_id=request.style_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Markdown parsing error: {str(e)}")
    
    try:
        hand = Hand()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize handwriting model: {str(e)}")
    
    # Create an indexed list of items to track the original order.
    indexed_items = []
    for i, (line, meta, bias, style, sw, sc) in enumerate(zip(lines, metadata, biases, styles, stroke_widths, stroke_colors)):
        indexed_items.append({
            "index": i,
            "line": line,
            "metadata": meta,
            "bias": bias,
            "style": style,
            "stroke_width": sw,
            "stroke_color": sc
        })

    # Reverse the indexed list for processing (to maintain alignment as needed)
    indexed_items_reversed = indexed_items[::-1]

    # Extract reversed lists for sampling
    rev_lines = [item["line"] for item in indexed_items_reversed]
    rev_biases = [item["bias"] for item in indexed_items_reversed]
    rev_styles = [item["style"] for item in indexed_items_reversed]

    try:
        # Generate stroke offsets for each reversed line using your GPU-accelerated model
        strokes = hand._sample(rev_lines, biases=rev_biases, styles=rev_styles)
        # for i, s in enumerate(strokes):
        #     print(f"Stroke {i} shape: {s.shape if hasattr(s, 'shape') else len(s)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during handwriting synthesis: {str(e)}")
    
    # Use a fixed line height and compute an independent starting coordinate per line
    LINE_HEIGHT = 50
    base_initial_coord = np.array([0, -(3 * LINE_HEIGHT / 4)])

    # Process each stroke concurrently using ThreadPoolExecutor.
    strokes_output = []
    with ThreadPoolExecutor() as executor:
        futures = []
        # Compute a per-line initial coordinate based on the line's index in the reversed order.
        for idx, (item, stroke) in enumerate(zip(indexed_items_reversed, strokes)):
            # Calculate starting coordinate for this line:
            initial_coord = base_initial_coord.copy()
            initial_coord[1] -= idx * LINE_HEIGHT
            futures.append(executor.submit(process_stroke, item, stroke, initial_coord))
        for future in futures:
            strokes_output.append(future.result())

    # Sort the processed output by the original index (to restore the original document order)
    strokes_output_sorted = sorted(strokes_output, key=lambda x: x["index"])

    # Optionally remove the index field from the final output
    for entry in strokes_output_sorted:
        del entry["index"]

    return {"strokes": strokes_output_sorted}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
