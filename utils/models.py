from pydantic import BaseModel
from typing import Optional, List

class MarkdownRequest(BaseModel):
    markdown: str
    style_id: Optional[int] = 8  # Default style id if not provided
    ref_strokes: Optional[list] = None
    screen_width: Optional[int] = 800  # Default screen width
    screen_height: Optional[int] = 600  # Default screen height

class WordRequest(BaseModel):
    text: str
    style_id: Optional[int] = 8  # Default style id if not provided
    ref_strokes: Optional[list] = None