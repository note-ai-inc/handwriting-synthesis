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


# --- Main ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Gemini service on port {port} using model {MODEL_NAME}")
    app.run(host="0.0.0.0", port=port, debug=False)