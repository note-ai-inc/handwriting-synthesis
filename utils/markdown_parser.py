import re

def parse_markdown(markdown_text):
    """
    Parses a markdown string and returns:
      - a list of text lines (with markdown markers removed and normalized)
      - a list of metadata dictionaries corresponding to each line with enhanced formatting info.
    """
    SMART_QUOTES = {
        """: '"', """: '"', "'": "'",
        "'": "'", "–": "-", "—": "-"
    }
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.*)")
    BULLET_PATTERN = re.compile(r"^(\s*)-\s+(.*)")  # Capture leading spaces for nesting
    NUMBERED_PATTERN = re.compile(r"^(\s*)\d+\.\s+(.*)")  # Capture leading spaces for nesting
    BLOCKQUOTE_PATTERN = re.compile(r"^(\s*)>\s+(.*)")  # Capture leading spaces for nesting
    EMPHASIS_BOLD = re.compile(r"(\*\*|__)")
    EMPHASIS_ITALIC = re.compile(r"(\*|_)")

    results = []
    line_meta_base = {
        "type": "paragraph", 
        "indent": 0, 
        "header_level": 0,
        "is_list_item": False,
        "list_type": None,
        "nesting_level": 0,
        "spacing_before": 0.5,  # Multiplier for spacing before this element
        "spacing_after": 0.5,   # Multiplier for spacing after this element
        "font_scale": 1.0       # Scale factor for text size
    }

    for line_idx, raw_line in enumerate(markdown_text.splitlines()):
        line = raw_line.strip()
        line_meta = line_meta_base.copy()

        # Assign each original raw line an ID
        line_meta["group_id"] = line_idx

        # Detect markdown constructs with enhanced metadata
        if line.startswith('#'):
            header_match = HEADER_PATTERN.match(line)
            if header_match:
                header_markers, line = header_match.groups()
                header_level = len(header_markers)
                line_meta["type"] = "header"
                line_meta["header_level"] = header_level
                
                # Header spacing and scaling based on level
                if header_level == 1:  # H1
                    line_meta["spacing_before"] = 0.6  # Reduced from 2.5
                    line_meta["spacing_after"] = 0.6   # Reduced from 1.8
                    line_meta["font_scale"] = 1.8
                elif header_level == 2:  # H2
                    line_meta["spacing_before"] = 0.5  # Reduced from 2.0
                    line_meta["spacing_after"] = 0.5   # Reduced from 1.5
                    line_meta["font_scale"] = 1.5
                elif header_level == 3:  # H3
                    line_meta["spacing_before"] = 0.5  # Reduced from 1.8
                    line_meta["spacing_after"] = 0.5   # Reduced from 1.3
                    line_meta["font_scale"] = 1.3
                elif header_level == 4:  # H4
                    line_meta["spacing_before"] = 0.5  # Reduced from 1.5
                    line_meta["spacing_after"] = 0.5   # Reduced from 1.2
                    line_meta["font_scale"] = 1.2
                elif header_level == 5:  # H5
                    line_meta["spacing_before"] = 0.5  # Reduced from 1.3
                    line_meta["spacing_after"] = 0.5   # Reduced from 1.1
                    line_meta["font_scale"] = 1.1
                else:  # H6
                    line_meta["spacing_before"] = 0.5  # Reduced from 1.2
                    line_meta["spacing_after"] = 0.5   # Same
                    line_meta["font_scale"] = 1.05
                    
        elif line.startswith('-') or (raw_line.startswith(' ') and '-' in raw_line):
            bullet_match = BULLET_PATTERN.match(raw_line)  # Use raw_line to capture indentation
            if bullet_match:
                leading_spaces, line = bullet_match.groups()
                nesting_level = len(leading_spaces) // 2  # 2 spaces per nesting level
                line_meta["type"] = "bullet"
                line_meta["is_list_item"] = True
                line_meta["list_type"] = "bullet"
                line_meta["nesting_level"] = nesting_level
                line_meta["indent"] = 1 + nesting_level
                line_meta["spacing_before"] = 0.8 if nesting_level == 0 else 0.6
                line_meta["spacing_after"] = 0.8 if nesting_level == 0 else 0.6
                
        elif line and (line[0].isdigit() or (raw_line.startswith(' ') and any(c.isdigit() for c in raw_line))):
            num_match = NUMBERED_PATTERN.match(raw_line)  # Use raw_line to capture indentation
            if num_match:
                leading_spaces, line = num_match.groups()
                nesting_level = len(leading_spaces) // 2  # 2 spaces per nesting level
                line_meta["type"] = "numbered"
                line_meta["is_list_item"] = True
                line_meta["list_type"] = "numbered"
                line_meta["nesting_level"] = nesting_level
                line_meta["indent"] = 1 + nesting_level
                line_meta["spacing_before"] = 0.8 if nesting_level == 0 else 0.6
                line_meta["spacing_after"] = 0.8 if nesting_level == 0 else 0.6
                
        elif line.startswith('>') or (raw_line.startswith(' ') and '>' in raw_line):
            blockquote_match = BLOCKQUOTE_PATTERN.match(raw_line)  # Use raw_line to capture indentation
            if blockquote_match:
                leading_spaces, line = blockquote_match.groups()
                nesting_level = len(leading_spaces) // 2
                line_meta["type"] = "blockquote"
                line_meta["nesting_level"] = nesting_level
                line_meta["indent"] = 1 + nesting_level
                line_meta["spacing_before"] = 1.2
                line_meta["spacing_after"] = 1.2
                line_meta["font_scale"] = 0.95  # Slightly smaller for quotes
        
        # Handle empty lines (paragraph breaks)
        elif not line.strip():
            line_meta["type"] = "empty"
            line_meta["spacing_before"] = 0.5
            line_meta["spacing_after"] = 0.5

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

        # 2) Wrap each final segment at responsive character limit based on screen width
        for segment in sub_lines:
            # Responsive character limit (will be passed from screen width)
            char_limit = 75  # Default, will be adjusted in calling function
            
            if len(segment) > 30:
                # Perform word-wrapping at char_limit
                words_for_wrap = segment.split()
                current_line = []
                current_length = 0

                for word in words_for_wrap:
                    if current_line and (current_length + len(word) + 1 > char_limit):
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