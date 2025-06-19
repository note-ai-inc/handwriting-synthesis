import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import threading
from model.train import StyleSynthesisModel
from model import drawing

# Thread lock for TensorFlow session safety
tf_lock = threading.RLock()

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
