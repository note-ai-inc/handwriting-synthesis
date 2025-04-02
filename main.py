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
    style_id: Optional[int] = 8  # Default style id if not provided

def parse_markdown(markdown_text):
    """
    Parses a markdown string and returns:
      - a list of text lines (with markdown markers removed and normalized)
      - a list of metadata dictionaries corresponding to each line.

    Steps included:
      - Identify markdown elements (headers, bullets, etc.) and store their metadata
      - Remove emphasis markers like ** or *
      - Split out any word > 7 chars into its own segment
      - Word-wrap at 75 chars
      - Remove or replace special characters if desired
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
    styles = [style_id for i in lines]

    pattern = [1, 1, 1, 1]
    stroke_widths = (pattern * ((n // len(pattern)) + 1))[:n]
    stroke_colors = [single_stroke_color] * n

    print(lines)
    print(biases)
    print(styles)
    print(stroke_widths)
    print(stroke_colors)
    
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

class Hand:
    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
                
                # Prime with style strokes and append target text
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
        samples = [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]
        return samples

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
    at a given 'initial_coord'. In this updated approach, we are 
    NOT forcibly offsetting each segment vertically by idx.
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

    list_of_segments: [
       [ [ (x,y), (x,y), ... ],  # stroke1 for sub-segment1
         [ (x,y), (x,y), ... ],  # stroke2 for sub-segment1
         ...
       ],
       [ [ (x,y), (x,y), ... ],  # stroke1 for sub-segment2
         ...
       ],
       ...
    ]
    Returns a single list of strokes with horizontal bounding-box alignment.
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

hand = Hand()

@app.get("/health")
def health():
    region = os.environ.get('CLOUD_RUN_REGION', 'unknown')
    vm_name = os.environ.get('HOSTNAME', 'unknown')
    return {
        "message": "OK",
        "region": region,
        "instance": os.environ.get('K_REVISION', 'unknown'),
        "vm": vm_name
    }

@app.get("/hello")
def hello():
    return {"message": "Hello, World!"}

@app.post("/convert")
def convert_markdown(request: MarkdownRequest):
    try:
        markdown_text = request.markdown
        lines, metadata = parse_markdown(markdown_text)
        print("lines", lines)
        print("metadata", metadata)
        
        joined_lines = "\n".join(lines)
        styles, biases, stroke_colors, stroke_widths = metadata_to_style(
            metadata, style_id=request.style_id, lines=joined_lines
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Style metadata error: {str(e)}")
    
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

    # Reverse for the model input
    indexed_items_reversed = indexed_items[::-1]
    rev_lines = [item["line"] for item in indexed_items_reversed]
    rev_biases = [item["bias"] for item in indexed_items_reversed]
    rev_styles = [item["style"] for item in indexed_items_reversed]

    # Call the handwriting model
    try:
        print("rev_lines", rev_lines)
        strokes = hand._sample(rev_lines, biases=rev_biases, styles=rev_styles)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during handwriting synthesis: {str(e)}")

    base_initial_coord = np.array([0, -30])

    strokes_output = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for idx, (item, stroke) in enumerate(zip(indexed_items_reversed, strokes)):
            initial_coord = base_initial_coord.copy()
            futures.append(executor.submit(process_stroke, item, stroke, initial_coord))

        for future in futures:
            strokes_output.append(future.result())

    # Re-sort to original order
    strokes_output_sorted = sorted(strokes_output, key=lambda x: x["index"])

    grouped_strokes = {}
    for entry in strokes_output_sorted:
        group_id = entry["metadata"]["group_id"]  # from parse_markdown
        if group_id not in grouped_strokes:
            grouped_strokes[group_id] = {
                "lines": [],
                "segments": [],
                "stroke_width": entry["stroke_width"],
                "stroke_color": entry["stroke_color"],
                "metadata": entry["metadata"],
            }
        grouped_strokes[group_id]["lines"].append(entry["line"])
        grouped_strokes[group_id]["segments"].append(entry["strokes"])

    merged_output = []
    for gid in sorted(grouped_strokes.keys()):
        group = grouped_strokes[gid]
        merged_line_text = " ".join(filter(None, group["lines"]))

        # chain all sub-segments horizontally
        combined_strokes = combine_segments_as_one_line(group["segments"], spacing=20)

        for stroke in combined_strokes:
            for pt in stroke:
                pt[1] += gid * 50

        indent = group["metadata"].get("indent", 0)
        if indent > 0:
            for stroke in combined_strokes:
                for pt in stroke:
                    pt[0] += indent * 50 + 20

        merged_output.append({
            "line": merged_line_text,
            "strokes": combined_strokes,
            "stroke_width": group["stroke_width"],
            "stroke_color": group["stroke_color"],
        })

    return {"strokes": merged_output}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
