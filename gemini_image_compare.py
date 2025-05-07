import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from google import genai  # UPDATED import to new client library
from google.genai import types  # ADDED for Part handling
from PIL import Image
from werkzeug.utils import secure_filename
import io  # ADDED for BytesIO handling

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found in environment variables")
    raise ValueError("GOOGLE_API_KEY environment variable is required")

# Initialize Gemini API (UPDATED to new client)
try:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    logger.info("Successfully configured Gemini client")
except Exception as e:
    logger.error(f"Failed to configure Gemini client: {e}")
    raise

# Define the model to be used
MODEL_NAME = "gemini-2.0-flash"

app = Flask(__name__)

# --- Health Check Endpoint ---
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(status="healthy"), 200

# --- Helper function to handle Gemini API response ---
def _process_gemini_response(response, model_name_for_log):
    try:
        # Check for safety issues first
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
            msg = response.prompt_feedback.block_reason_message or f"Reason code: {response.prompt_feedback.block_reason}"
            logger.warning(f"Prompt blocked for model {model_name_for_log}. Reason: {msg}")
            raise RuntimeError(f"Request failed due to content moderation: {msg}")

        # Access the text
        generated_text = response.text
        return generated_text

    except ValueError as e:
        logger.error(f"ValueError accessing response.text for model {model_name_for_log}: {e}. This can happen if the response was blocked for safety or if no text was generated.")
        raise RuntimeError(f"Gemini model {model_name_for_log} did not return valid text output. This might be due to safety filters or an issue with the generated content. Original error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error processing Gemini response for model {model_name_for_log}: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error processing Gemini response: {str(e)}")


# --- Handwriting Quality Check Function ---
def check_handwriting(image_path: str) -> bool:
    """
    Returns True if handwriting quality is good AND no misspelled words,
    False otherwise.
    """
    try:
        logger.debug(f"Opening image for handwriting check: {image_path}")
        
        # Read the image as bytes
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        
    except FileNotFoundError:
        logger.error(f"Image file not found at {image_path}")
        raise RuntimeError(f"Image file not found: {image_path}")
    except Exception as e:
        logger.error(f"Failed to open image {image_path}: {e}")
        raise RuntimeError(f"Image loading failed: {e}")

    prompt = (
        "You will be given a single handwriting image.\n"
        "1. Evaluate the overall handwriting quality. Is it 'good' or 'bad'?\n"
        "2. Check if any word is written incorrectly (misspelled or malformed).\n"
        "Based on these two points, if the quality is 'good' AND all words are correct, "
        "respond with exactly 'true'. Otherwise, respond with exactly 'false'. "
        "Do not add any other explanation or text."
    )

    try:
        logger.debug(f"Sending request to Gemini model {MODEL_NAME} for handwriting check.")
        
        # Create image part
        img_part = types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg')
        
        # Generate content using new client approach
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, img_part]
        )
        
        generated_text = _process_gemini_response(response, MODEL_NAME)
        txt = generated_text.lower().strip()
        
        logger.info(f"Gemini raw response for handwriting check: '{txt}'")
        if txt == "true":
            return True
        elif txt == "false":
            return False
        else:
            # If the model doesn't follow instructions perfectly, this will be logged.
            logger.error(f"Gemini response for handwriting check ('{txt}') was not strictly 'true' or 'false'.")
            raise RuntimeError(f"Unexpected response from Gemini for handwriting check: '{txt}'. Expected 'true' or 'false'.")

    except Exception as e:
        logger.error(f"Gemini API call or processing failed for handwriting check: {e}", exc_info=True)
        raise RuntimeError(f"Handwriting check with Gemini failed: {e}")


# --- Endpoint: /check_handwriting ---
@app.route("/check_handwriting", methods=["POST"])
def api_check_handwriting():
    logger.debug("Received /check_handwriting request")
    if "image" not in request.files:
        logger.error("Missing 'image' file in /check_handwriting request")
        return jsonify(ok=False, error="Missing 'image' file"), 400

    img_file = request.files["image"]
    
    original_filename = img_file.filename
    if not original_filename:
        logger.error("Empty filename in /check_handwriting request")
        return jsonify(ok=False, error="Empty filename"), 400
    
    safe_filename = secure_filename(original_filename)
    if not safe_filename: # if filename was e.g. ".." or just dots
        safe_filename = "uploaded_image_check" # provide a default
        
    temp_dir = "/tmp"
    if not os.path.exists(temp_dir):
        try:
            os.makedirs(temp_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create temp directory {temp_dir}: {e}")
            return jsonify(ok=False, error=f"Could not create temp directory: {e}"), 500

    temp_path = os.path.join(temp_dir, safe_filename)
    
    try:
        logger.debug(f"Saving image to temporary path: {temp_path}")
        img_file.save(temp_path)
        
        result = check_handwriting(temp_path)
        logger.info(f"Handwriting check result for {safe_filename}: {result}")
        return jsonify(ok=True, result=result)
    except RuntimeError as e: # Errors from our logic or Gemini communication
        logger.error(f"Runtime error during handwriting check for {safe_filename}: {e}")
        return jsonify(ok=False, error=str(e)), 502 # 502 Bad Gateway if our call to Gemini fails
    except Exception as e: # Other unexpected errors
        logger.error(f"Unexpected error during handwriting check for {safe_filename}: {e}", exc_info=True)
        return jsonify(ok=False, error=f"An unexpected server error occurred: {str(e)}"), 500
    finally:
        try:
            if os.path.exists(temp_path):
                # os.remove(temp_path)
                logger.debug(f"Cleaned up temporary file: {temp_path}")
        except OSError as e:
            logger.error(f"Error cleaning up temporary file {temp_path}: {e}")


# --- Handwriting Text Extraction Function ---
def extract_text_from_image(image_path: str) -> str:
    """
    Extracts only the handwritten text from the image with no extra commentary.
    """
    try:
        logger.debug(f"Opening image for text extraction: {image_path}")
        
        # Read the image as bytes
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
            
    except FileNotFoundError:
        logger.error(f"Image file not found at {image_path}")
        raise RuntimeError(f"Image file not found: {image_path}")
    except Exception as e:
        logger.error(f"Failed to open image {image_path}: {e}")
        raise RuntimeError(f"Image loading failed: {e}")

    prompt = (
        "You will be given a handwriting image.\n"
        "Your task is to extract and return only the handwritten text exactly as it appears.\n"
        "Do not add any explanation, commentary, or formatting.\n"
        "Output only the text from the image. Nothing else."
    )
    
    try:
        logger.debug(f"Sending request to Gemini model {MODEL_NAME} for text extraction.")
        
        # Create image part
        img_part = types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg')
        
        # Generate content using new client approach
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, img_part]
        )
        
        extracted_text = _process_gemini_response(response, MODEL_NAME)
        
        logger.info(f"Successfully extracted text (first 100 chars): '{extracted_text[:100]}...'")
        return extracted_text.strip() # Ensure any leading/trailing whitespace from model is stripped

    except Exception as e:
        logger.error(f"Gemini API call or processing failed for text extraction: {e}", exc_info=True)
        raise RuntimeError(f"Text extraction with Gemini failed: {e}")


# --- Endpoint: /extract_text ---
@app.route("/extract_text", methods=["POST"])
def api_extract_text():
    logger.debug("Received /extract_text request")
    
    if "image" not in request.files:
        logger.error("Missing 'image' file in /extract_text request")
        return jsonify(ok=False, error="Missing 'image' file"), 400

    img_file = request.files["image"]

    original_filename = img_file.filename
    if not original_filename:
        logger.error("Empty filename in /extract_text request")
        return jsonify(ok=False, error="Empty filename"), 400

    safe_filename = secure_filename(original_filename)
    if not safe_filename:
        safe_filename = "uploaded_image_extract"

    temp_dir = "/tmp"
    if not os.path.exists(temp_dir):
        try:
            os.makedirs(temp_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create temp directory {temp_dir}: {e}")
            return jsonify(ok=False, error=f"Could not create temp directory: {e}"), 500
            
    temp_path = os.path.join(temp_dir, safe_filename)

    try:
        logger.debug(f"Saving image to temporary path: {temp_path}")
        img_file.save(temp_path)
        
        logger.debug(f"Extracting text from image {safe_filename}")
        extracted_text = extract_text_from_image(temp_path)
        logger.info(f"Successfully extracted text from {safe_filename}")
        
        return jsonify(ok=True, text=extracted_text)
    except RuntimeError as e:
        logger.error(f"Runtime error during text extraction for {safe_filename}: {e}")
        return jsonify(ok=False, error=str(e)), 502
    except Exception as e:
        logger.error(f"Unexpected error during text extraction for {safe_filename}: {e}", exc_info=True)
        return jsonify(ok=False, error=f"An unexpected server error occurred: {str(e)}"), 500
    finally:
        try:
            if os.path.exists(temp_path):
                # os.remove(temp_path)
                logger.debug(f"Cleaned up temporary file: {temp_path}")
        except OSError as e:
            logger.error(f"Error cleaning up temporary file {temp_path}: {e}")


# --- Multiple Image Comparison Function (NEW) ---
def compare_images(image_paths: list) -> str:
    """
    Compares multiple images and returns the differences described by Gemini.
    """
    try:
        # Prepare content parts list starting with the prompt
        contents = ["What is different between these images?"]
        
        # Add each image as a part
        for idx, img_path in enumerate(image_paths):
            logger.debug(f"Processing image {idx+1} at path: {img_path}")
            
            # Read image file as bytes
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
                
            # For first image, we could optionally use file upload API
            # But for simplicity, we'll use the same approach for all images
            img_part = types.Part.from_bytes(
                data=img_bytes, 
                mime_type='image/jpeg'  # Adjust mime type if needed
            )
            
            contents.append(img_part)
            
        # Generate content with multiple images
        logger.debug(f"Sending request to Gemini model {MODEL_NAME} for image comparison.")
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents
        )
        
        comparison_text = _process_gemini_response(response, MODEL_NAME)
        logger.info(f"Successfully compared images, result (first 100 chars): '{comparison_text[:100]}...'")
        
        return comparison_text.strip()
        
    except Exception as e:
        logger.error(f"Gemini API call or processing failed for image comparison: {e}", exc_info=True)
        raise RuntimeError(f"Image comparison with Gemini failed: {e}")


# --- Endpoint: /compare_images (NEW) ---
@app.route("/compare_images", methods=["POST"])
def api_compare_images():
    logger.debug("Received /compare_images request")
    
    if not request.files or "images[]" not in request.files:
        logger.error("Missing 'images[]' files in /compare_images request")
        return jsonify(ok=False, error="Missing images. Please send multiple images with key 'images[]'"), 400
    
    # Get list of image files
    img_files = request.files.getlist("images[]")
    
    if len(img_files) < 2:
        logger.error(f"Not enough images provided: {len(img_files)}")
        return jsonify(ok=False, error="At least two images are required for comparison"), 400
    
    temp_dir = "/tmp"
    if not os.path.exists(temp_dir):
        try:
            os.makedirs(temp_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create temp directory {temp_dir}: {e}")
            return jsonify(ok=False, error=f"Could not create temp directory: {e}"), 500
    
    temp_paths = []
    
    try:
        # Save all images to temporary files
        for idx, img_file in enumerate(img_files):
            original_filename = img_file.filename or f"compare_image_{idx}"
            safe_filename = secure_filename(original_filename)
            if not safe_filename:
                safe_filename = f"compare_image_{idx}"
                
            temp_path = os.path.join(temp_dir, safe_filename)
            logger.debug(f"Saving image {idx+1} to temporary path: {temp_path}")
            img_file.save(temp_path)
            temp_paths.append(temp_path)
        
        # Compare the images
        comparison_result = compare_images(temp_paths)
        logger.info(f"Successfully compared {len(temp_paths)} images")
        
        return jsonify(ok=True, comparison=comparison_result)
    except RuntimeError as e:
        logger.error(f"Runtime error during image comparison: {e}")
        return jsonify(ok=False, error=str(e)), 502
    except Exception as e:
        logger.error(f"Unexpected error during image comparison: {e}", exc_info=True)
        return jsonify(ok=False, error=f"An unexpected server error occurred: {str(e)}"), 500
    finally:
        # Clean up temporary files
        for temp_path in temp_paths:
            try:
                if os.path.exists(temp_path):
                    # os.remove(temp_path)
                    logger.debug(f"Cleaned up temporary file: {temp_path}")
            except OSError as e:
                logger.error(f"Error cleaning up temporary file {temp_path}: {e}")


# --- Main ---
if __name__ == "__main__":
    logger.info(f"Starting Gemini service on port 5000 using model {MODEL_NAME}")
    # For development, debug=True can be useful. For production, use a WSGI server.
    app.run(host="0.0.0.0", port=5000, debug=False)