# Handwriting Synthesis API - Integration Guide

## API Endpoint
```
POST /convert
```

## Request Format
```typescript
interface MarkdownRequest {
    markdown: string;          // The text to convert to handwriting
    style_id?: number;         // Optional: Predefined style ID (default: 8)
    ref_strokes?: Stroke[];    // Optional: Custom reference strokes for style
}

interface Stroke {
    x: float;      // x-coordinate (normalized between -1 and 1)
    y: float;      // y-coordinate (normalized between -1 and 1)
    eos: number;   // end-of-stroke marker (1 for start of stroke, 0 for continuation)
}
```

## Example Request
```json
{
    "markdown": "Hello World",
    "ref_strokes": [
        // First stroke
        {"x": 0.0, "y": 0.0, "eos": 1},        // Start of first stroke
        {"x": -0.011367, "y": 0.05825589, "eos": 0},   // Middle of stroke
        {"x": -0.03552188, "y": 0.12929966, "eos": 0},   // Middle of stroke
        {"x": -0.06393939, "y": 0.17760941, "eos": 0},   // Middle of stroke
        {"x": -0.12219528, "y": 0.22165656, "eos": 0},   // End of first stroke
        
        // Second stroke
        {"x": -0.13640404, "y": 0.2017643, "eos": 1},   // Start of second stroke
        {"x": -0.2585993, "y": 0.24296969, "eos": 0},   // Middle of stroke
        {"x": -0.3239596, "y": 0.26854545, "eos": 0},   // Middle of stroke
        {"x": -0.41205385, "y": 0.30406734, "eos": 0}    // End of second stroke
    ]
}
```

## Response Format
```typescript
interface Response {
    strokes: StrokeGroup[];
}

interface StrokeGroup {
    line: string;              // The text line
    strokes: Point[][];        // Array of strokes, each stroke is array of points
    stroke_width: number;      // Width of the stroke
    stroke_color: string;      // Color of the stroke
}

interface Point {
    x: float;     // x-coordinate
    y: float;     // y-coordinate
}
```

## Important Notes

1. **Stroke Data Format**:
   - Each point must have x, y coordinates and an eos marker
   - x and y coordinates must be float values
   - Coordinates must be normalized between -1 and 1
   - eos = 1 indicates start of a new stroke
   - eos = 0 indicates continuation of the current stroke
   - Strokes should be in the order they were written
   - Typical stroke data contains 400-500 points

2. **Coordinate System**:
   - x-coordinates (float) typically range from -1.3 to 0
   - y-coordinates (float) typically range from -0.6 to 0.9
   - All coordinates should be normalized to this range
   - Maintain relative proportions when normalizing
   - Use float precision for coordinates (e.g., -0.011367, 0.05825589)

3. **Stroke Collection Guidelines**:
   - Collect points continuously as the user writes
   - Mark the start of each new stroke with eos = 1
   - Mark all other points with eos = 0
   - Maintain the natural writing order
   - Store points in sequence as they are drawn

4. **Implementation Tips**:
   - Normalize coordinates before sending to API
   - Use a data structure to store points in sequence
   - Track the current stroke and all completed strokes
   - Clear stroke data after successful API calls
   - Implement proper error handling

5. **Error Handling**:
   - Handle API errors appropriately
   - Validate stroke data before sending
   - Provide feedback to users if stroke collection fails
   - Check for network connectivity
   - Handle timeouts and retries

6. **Best Practices**:
   - Collect strokes in real-time as the user writes
   - Maintain stroke order and timing
   - Clear stroke data after successful API calls
   - Implement proper error handling and user feedback
   - Consider implementing a stroke preview feature
   - Optimize network calls by batching strokes if needed
   - Cache successful results when appropriate

## CORS Configuration
The API is configured to accept requests from:
- http://127.0.0.1:5500
- http://localhost:5500
- All origins (*)

Make sure your application's origin is included in the allowed origins list.


# Dummy line
