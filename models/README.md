# LLM Models Comparison

## OpenAI (GPT-4, GPT-3.5)

### Input Data Types
- Text: Plain text in messages
- Images: Base64 or URL in content array with type "image_url"
- Files are not directly supported in the API

```python
messages=[{
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "What's in this image?"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,..."
            }
        }
    ]
}]
```

### Function Calling
Two approaches available:
1. Legacy functions:
```python
functions=[{
    "name": "get_weather",
    "description": "Get weather in location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        }
    }
}]
function_call="auto"  # or {"name": "get_weather"}
```

2. New tools approach:
```python
tools=[{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather in location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            }
        }
    }
}]
tool_choice="auto"  # or {"type": "function", "function": {"name": "get_weather"}}
```

### Response Formats
1. Text (default):
```python
response_format={"type": "text"}
```

2. JSON:
```python
response_format={"type": "json_object"}
```

### Response Types
1. HTTP Response:
- Single response with complete message
- Contains usage statistics
- Includes all tool/function calls at once

2. Streaming:
- Chunks of content via delta updates
- Tool/function calls can come in chunks
- Final chunk includes finish_reason
- Usage statistics only in non-streaming mode

## Anthropic (Claude)

### Input Data Types
- Text: Plain text or in content array
- Images: Base64 in content array with type "image"
- PDF: Base64 in content array with type "file"
- Maximum size: 100MB per file

```python
content=[
    {
        "type": "text",
        "text": "Analyze this document"
    },
    {
        "type": "file",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": "base64_encoded_content..."
        }
    }
]
```

### Function Calling
Only tools approach (no legacy functions):
```python
system_tools=[{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather in location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            }
        }
    }
}]
```
- No direct way to force specific function call
- Tool responses come in tool_calls field

### Response Formats
- Only text format supported
- No native JSON mode (though Claude can format responses as JSON if asked)

### Response Types
1. HTTP Response:
- Complete message with all content
- Includes usage statistics
- Tool calls included in response

2. Streaming:
- More detailed streaming events:
  * message_start
  * content_block_start
  * content_block_delta
  * content_block_stop
  * message_delta
  * message_stop
- Usage statistics available in final message_stop event

## Google (Gemini)

### Input Data Types
- Text: Plain text in parts
- Images: Base64 (inline_data) or URI (file_data)
- Video: MP4, MOV via URI
- Audio: AAC, WAV, MP3, OGG via URI

```python
contents=[{
    "parts": [
        {
            "text": "What's in this image?"
        },
        {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": "base64_encoded_content..."
            }
        }
        # Or using URI
        {
            "file_data": {
                "file_uri": "gs://bucket/video.mp4",
                "mime_type": "video/mp4"
            }
        }
    ]
}]
```

### Function Calling
Single tools approach with function declarations:
```python
tools=[{
    "function_declarations": [{
        "name": "get_weather",
        "description": "Get weather in location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            }
        }
    }]
}]
```
- Function calls come in structured format
- Supports function response injection

### Response Formats
- Primary text format
- Can return JSON through prompt engineering
- No native JSON mode

### Response Types
1. HTTP Response:
- Complete response with all candidates
- Includes safety ratings
- Citation metadata when available
- Token count per response

2. Streaming:
- Streams candidates progressively
- Safety ratings per chunk
- Usage metadata in final chunk
- Supports citation streaming

### Unique Features
- Built-in citation system
- Detailed safety controls
- Video/Audio support
- Multiple candidate responses
- URI-based file loading