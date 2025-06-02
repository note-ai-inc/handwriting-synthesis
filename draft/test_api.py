import requests

# API base URL
BASE_URL = "http://localhost:5000"

def test_handwriting():
    # Path to your test image
    image_path = "strokes.png"
    
    try:
        # Open and send the image
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{BASE_URL}/check_handwriting", files=files)
        
        # Print the response
        print("\nHandwriting Check Response:", response.json())
        
    except Exception as e:
        print("Error:", str(e))

def test_extract_text():
    # Path to your test image
    image_path = "strokes.png"
    
    try:
        # Open and send the image
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{BASE_URL}/extract_text", files=files)
        
        # Print the response
        print("\nText Extraction Response:", response.json())
        
    except Exception as e:
        print("Error:", str(e))

def test_compare_images():
    # Path to your test image
    image_path = "strokes.png"
    
    try:
        # Open and send the same image twice for comparison
        with open(image_path, 'rb') as f1, open(image_path, 'rb') as f2:
            files = [
                ('images[]', f1),
                ('images[]', f2)
            ]
            response = requests.post(f"{BASE_URL}/compare_images", files=files)
        
        # Print the response
        print("\nImage Comparison Response:", response.json())
        
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    print("Testing all APIs...")
    test_handwriting()
    test_extract_text()
    test_compare_images() 