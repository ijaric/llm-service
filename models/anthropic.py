"""Anthropic Claude API Models

This module provides Pydantic models for Anthropic's Claude API.

Key Features:
1. Data Types Support:
   - Text: Regular text or structured content array
   - Images: Base64 encoded with type "image"
   - PDF: Base64 encoded with type "file"
   - Size limit: 100MB per file
   
2. Function Calling:
   - Only new tools approach (no legacy functions)
   - No direct way to force specific function call
   - Tool responses in tool_calls field
   
3. Response Formats:
   - Only text format supported
   - No native JSON mode (but can format as JSON through prompting)
   
4. Response Types:
   - HTTP: Complete message with usage stats
   - Streaming: Detailed events (message_start, content_block_*, message_stop)

Example Usage:
    ```python
    # File input (PDF)
    message = Message(
        role="user",
        content=[
            TextContent(
                type="text",
                text="Analyze this document"
            ),
            FileContent(
                type="file",
                source=MediaSource(
                    type="base64",
                    media_type="application/pdf",
                    data="base64_encoded_content..."
                )
            )
        ]
    )

    # Function calling
    request = MessageRequest(
        model="claude-3",
        messages=[message],
        system_tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather in location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        }]
    )
    ```

Streaming Response Events:
1. message_start: Initial message metadata
2. content_block_start: Start of new content block
3. content_block_delta: Content updates
4. content_block_stop: End of content block
5. message_delta: Updates to message metadata
6. message_stop: Final message with usage stats
"""

import typing
import pydantic

class MediaSource(pydantic.BaseModel):
    type: typing.Literal["base64"]
    media_type: str
    data: str

class ImageContent(pydantic.BaseModel):
    type: typing.Literal["image"]
    source: MediaSource

class TextContent(pydantic.BaseModel):
    type: typing.Literal["text"]
    text: str

class FileContent(pydantic.BaseModel):
    type: typing.Literal["file"]
    source: MediaSource

class FunctionParameters(pydantic.BaseModel):
    type: str
    description: str | None = None
    enum: list[str] | None = None
    items: dict[str, typing.Any] | None = None
    properties: dict[str, 'FunctionParameters'] | None = None
    required: list[str] | None = None

class Function(pydantic.BaseModel):
    name: str
    description: str
    parameters: FunctionParameters

class Tool(pydantic.BaseModel):
    type: typing.Literal["function"]
    function: Function

class Message(pydantic.BaseModel):
    role: typing.Literal["user", "assistant"]
    content: str | list[TextContent | ImageContent | FileContent]
    tool_calls: list[dict[str, typing.Any]] | None = None

class MessageRequest(pydantic.BaseModel):
    model: str
    messages: list[Message]
    system: str | None = None
    max_tokens: int | None = None
    metadata: dict[str, typing.Any] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    system_tools: list[Tool] | None = None

class Usage(pydantic.BaseModel):
    input_tokens: int
    output_tokens: int

class ContentBlock(pydantic.BaseModel):
    type: typing.Literal["text", "image", "file"]
    text: str | None = None
    source: MediaSource | None = None

class MessageResponse(pydantic.BaseModel):
    id: str
    type: str
    role: str
    content: list[ContentBlock]
    model: str
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: Usage
    tool_calls: list[dict[str, typing.Any]] | None = None

class Delta(pydantic.BaseModel):
    type: str | None = None
    text: str | None = None
    tool_calls: list[dict[str, typing.Any]] | None = None
    stop_reason: str | None = None
    stop_sequence: str | None = None

class StreamMessage(pydantic.BaseModel):
    type: str
    delta: Delta
    usage: Usage | None = None
    index: int = 0

class MessageStreamResponse(pydantic.BaseModel):
    id: str
    type: typing.Literal["message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
    message: StreamMessage
    role: str | None = None
    content_block: ContentBlock | None = None