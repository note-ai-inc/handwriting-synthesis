# Handwriting Synthesis API

A FastAPI-based service that converts markdown text into realistic handwritten strokes using deep learning. This application uses a neural network model to generate handwriting that mimics human writing patterns, with support for custom styles and responsive layout.

## Features

### Core Functionality
- **Markdown to Handwriting**: Converts markdown text into handwritten stroke sequences
- **Style Synthesis**: Uses pre-trained neural network models to generate realistic handwriting
- **Custom Style Support**: Accepts reference strokes to mimic specific handwriting styles
- **Responsive Layout**: Adapts text layout based on screen dimensions
- **Quality Control**: Integrated handwriting quality assessment using external service

### Markdown Support
- **Headers**: H1-H6 with appropriate sizing and spacing
- **Lists**: Bullet points and numbered lists with proper nesting
- **Blockquotes**: Indented quote formatting
- **Text Formatting**: Handles bold, italic, and smart quotes
- **Line Wrapping**: Intelligent text wrapping based on screen width
- **Indentation**: Proper indentation for different content types

### Advanced Features
- **Parallel Processing**: Multi-threaded stroke generation for improved performance
- **Deterministic Output**: Consistent results with seed-based randomization
- **Bounds Management**: Ensures all strokes fit within canvas boundaries
- **Dynamic Spacing**: Context-aware line spacing based on content type

## Architecture

### Core Components

1. **StyleSynthesisModel**: Neural network model for handwriting generation
   - LSTM-based architecture with attention mechanism
   - Style embedding for consistent handwriting characteristics
   - Mixture density networks for stroke prediction

2. **Markdown Parser**: Converts markdown to structured text with metadata
   - Extracts formatting information (headers, lists, quotes)
   - Calculates indentation and spacing requirements
   - Handles text wrapping and line breaks

3. **Stroke Processor**: Converts text to handwriting coordinates
   - Transforms neural network output to drawable strokes
   - Applies responsive scaling and positioning
   - Manages indentation and layout

4. **Quality Control**: External service integration for handwriting assessment
   - Validates generated handwriting quality
   - Falls back to default styles if quality is poor
   - Prevents infinite recursion in quality checks

### API Endpoints

#### `POST /convert`
Main endpoint for converting markdown to handwriting strokes.

**Request Body:**
```json
{
  "markdown": "# Hello World\nThis is a test.",
  "style_id": 8,
  "ref_strokes": null,
  "screen_width": 800,
  "screen_height": 600
}
```

**Response:**
```json
{
  "strokes": [
    {
      "line": "Hello World",
      "strokes": [[[x1, y1], [x2, y2], ...]],
      "stroke_width": 1,
      "stroke_color": "black",
      "metadata": {...}
    }
  ]
}
```

#### `GET /health`
Health check endpoint returning service status and instance information.

#### `GET /hello`
Simple test endpoint returning a greeting message.

## Technical Details

### Neural Network Model
- **Architecture**: LSTM with attention mechanism
- **Input**: Character sequences and style reference strokes
- **Output**: Stroke coordinates with end-of-stroke markers
- **Training**: Pre-trained on handwriting datasets
- **Inference**: Deterministic sampling with temperature control

### Coordinate System
- **Stroke Format**: `[x, y, end_of_stroke]` triplets
- **Coordinate Space**: Normalized to screen dimensions
- **Bounds Checking**: Automatic scaling to fit canvas
- **Precision**: 4 decimal places for coordinate accuracy

### Performance Optimizations
- **Thread Pool**: Parallel processing of text lines
- **Session Management**: Shared TensorFlow session across threads
- **Memory Management**: Efficient tensor allocation and cleanup
- **Caching**: Style data loaded once and reused

### Responsive Design
- **Screen Adaptation**: Layout adjusts to screen dimensions
- **Font Scaling**: Headers scale appropriately for different sizes
- **Indentation**: Responsive indentation based on screen width
- **Line Spacing**: Dynamic spacing based on content type and screen height

## Dependencies

### Core Libraries
- **FastAPI**: Web framework for API endpoints
- **TensorFlow**: Neural network inference
- **NumPy**: Numerical computations
- **Matplotlib**: Stroke rendering and visualization
- **Pillow**: Image processing

### Model Components
- **train.py**: Contains StyleSynthesisModel and StyleDataReader classes
- **drawing.py**: Utility functions for coordinate processing
- **Style Files**: Pre-trained style embeddings and character mappings

### External Services
- **Handwriting Quality Service**: External API for quality assessment
- **CORS Support**: Cross-origin request handling for web clients

## Configuration

### Environment Variables
- `GEMINI_SERVICE_URL`: URL for handwriting quality service (default: http://handwriting-quality:5000)
- `PORT`: Server port (default: 8080)
- `CLOUD_RUN_REGION`: Cloud deployment region
- `HOSTNAME`: Instance hostname

### Model Configuration
- **Checkpoint Path**: `checkpoints/model-10350`
- **LSTM Size**: 400 units
- **Attention Components**: 10 mixture components
- **Output Components**: 20 mixture components
- **Style Embedding Size**: 256 dimensions

### Processing Parameters
- **Temperature**: 0.5 (controls randomness)
- **Max Steps**: 40 × text length
- **Batch Size**: 32
- **Thread Count**: 2 × CPU cores (max 16)

## Usage Examples

### Basic Markdown Conversion
```python
import requests

response = requests.post("http://localhost:8080/convert", json={
    "markdown": "# My Document\n\nThis is a paragraph with **bold** text."
})
strokes = response.json()["strokes"]
```

### Custom Style with Reference Strokes
```python
ref_strokes = [
    {"x": 0, "y": 0, "eos": 0},
    {"x": 10, "y": 5, "eos": 0},
    {"x": 20, "y": 0, "eos": 1}
]

response = requests.post("http://localhost:8080/convert", json={
    "markdown": "Custom handwriting style",
    "ref_strokes": ref_strokes,
    "screen_width": 1200,
    "screen_height": 800
})
```

### Responsive Layout
```python
# Mobile layout
mobile_response = requests.post("http://localhost:8080/convert", json={
    "markdown": "Mobile-optimized text",
    "screen_width": 375,
    "screen_height": 667
})

# Desktop layout
desktop_response = requests.post("http://localhost:8080/convert", json={
    "markdown": "Desktop-optimized text",
    "screen_width": 1920,
    "screen_height": 1080
})
```

## Error Handling

### Quality Control
- Automatic fallback to default styles if quality check fails
- Prevents infinite recursion in quality assessment
- Graceful degradation when quality service is unavailable

### Processing Errors
- Thread-safe error handling in parallel processing
- Fallback to sequential processing if thread pool fails
- Placeholder generation for failed individual items

### Input Validation
- Markdown parsing error handling
- Coordinate bounds validation
- Reference stroke normalization

## Deployment

### Local Development
```bash
pip install -r requirements.txt
python main.py
```

### Docker Deployment
```dockerfile
FROM python:3.8-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
```

### Cloud Run Deployment
- Supports Google Cloud Run with automatic scaling
- Health check endpoint for load balancer integration
- Instance metadata reporting for monitoring

## Performance Considerations

### Scalability
- Stateless design for horizontal scaling
- Shared model loading across requests
- Efficient memory usage with tensor reuse

### Latency Optimization
- Parallel processing reduces response time
- Pre-loaded models eliminate startup delays
- Optimized coordinate processing pipeline

### Resource Management
- Configurable thread pool sizing
- Memory-efficient stroke storage
- Automatic cleanup of temporary files

## Monitoring and Logging

### Health Monitoring
- `/health` endpoint for service status
- Instance and region information
- Error rate tracking through logs

### Performance Metrics
- Processing time per request
- Thread utilization statistics
- Memory usage monitoring

### Debug Information
- Detailed logging for processing steps
- Error context preservation
- Quality check result tracking
