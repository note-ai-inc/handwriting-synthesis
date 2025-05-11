# Handwriting Synthesis and Quality Analysis Services

This repository contains two microservices that work together to provide handwriting synthesis and quality analysis capabilities:

1. **Handwriting Synthesis Service** (`main.py`)
2. **Handwriting Quality Analysis Service** (`gemini_image_compare.py`)

## Overview

### Handwriting Synthesis Service

The Handwriting Synthesis Service converts markdown text into handwritten strokes using a machine learning model. It supports both default style-based generation and reference stroke-based generation.

**Key Features:**
- Converts markdown text to handwritten strokes
- Supports multiple handwriting styles
- Maintains markdown formatting (headers, lists, indentation)
- Parallel processing for better performance
- Quality control with reference strokes

**API Endpoints:**
- `POST /convert`: Converts markdown to handwritten strokes
  - Input: Markdown text, style ID, and optional reference strokes
  - Output: Sequence of handwritten strokes
- `GET /health`: Health check endpoint
- `GET /hello`: Simple test endpoint

### Handwriting Quality Analysis Service

The Handwriting Quality Analysis Service uses Google's Gemini AI model to analyze handwriting quality and extract text from handwritten images.

**Key Features:**
- Handwriting quality assessment
- Text extraction from handwritten images
- Multiple image comparison
- Integration with Google's Gemini AI

**API Endpoints:**
- `POST /check_handwriting`: Evaluates handwriting quality
  - Input: Handwritten image
  - Output: Quality assessment (true/false)
- `POST /extract_text`: Extracts text from handwritten images
  - Input: Handwritten image
  - Output: Extracted text
- `POST /compare_images`: Compares multiple handwritten images
  - Input: Multiple handwritten images
  - Output: Comparison analysis
- `GET /health`: Health check endpoint

## Setup and Installation

### Prerequisites
- Python 3.7+
- TensorFlow
- FastAPI
- Flask
- Google Cloud credentials (for Gemini API)
- Other dependencies listed in requirements.txt

### Environment Variables
```bash
# Required for Gemini service
GOOGLE_API_KEY=your_google_api_key

# Optional configurations
PORT=8080  # For handwriting synthesis service
GEMINI_SERVICE_URL=http://handwriting-quality:5000  # URL for quality service
```

### Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Services

### Handwriting Synthesis Service
```bash
python main.py
```
The service will start on port 8080 by default.

### Handwriting Quality Analysis Service
```bash
python gemini_image_compare.py
```
The service will start on port 5000 by default.

## API Usage Examples

### Converting Markdown to Handwriting
```python
import requests

response = requests.post(
    "http://localhost:8080/convert",
    json={
        "markdown": "# Hello World\nThis is a test",
        "style_id": 8,
        "ref_strokes": None  # Optional reference strokes
    }
)
```

### Checking Handwriting Quality
```python
import requests

with open("handwriting.png", "rb") as f:
    response = requests.post(
        "http://localhost:5000/check_handwriting",
        files={"image": f}
    )
```

### Extracting Text from Handwriting
```python
import requests

with open("handwriting.png", "rb") as f:
    response = requests.post(
        "http://localhost:5000/extract_text",
        files={"image": f}
    )
```

## Architecture

The system uses a microservices architecture:
1. The Handwriting Synthesis Service handles the conversion of text to handwriting
2. The Handwriting Quality Analysis Service provides quality assessment and text extraction
3. Services communicate via HTTP APIs
4. The Gemini AI model is used for quality analysis and text extraction

## Error Handling

Both services implement comprehensive error handling:
- Input validation
- File processing errors
- API communication errors
- Model inference errors
- Proper error responses with status codes

## Logging

Both services implement detailed logging:
- Request/response logging
- Error logging
- Performance metrics
- Debug information

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
