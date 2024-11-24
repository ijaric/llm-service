"""Google Gemini API Models

This module provides Pydantic models for Google's Gemini API.

Key Features:
1. Data Types Support:
   - Text: Plain text in parts
   - Images: Base64 (inline_data) or URI (file_data)
   - Video: MP4, MOV via URI
   - Audio: AAC, WAV, MP3, OGG via URI
   
2. Function Calling:
   - Single tools approach with function declarations
   - Structured function calls and responses
   - Support for function response injection
   
3. Response Formats:
   - Primary text format
   - JSON through prompt engineering
   - No native JSON mode
   
4. Response Types:
   - HTTP: Complete response with candidates
   - Streaming: Progressive with safety ratings

Unique Features:
- Built-in citation system
- Detailed safety controls
- Video/Audio support
- Multiple candidate responses
- URI-based file loading

Example Usage:
    ```python
    # Multimodal input
    content = Content(
        parts=[
            Part(text="What's in this video?"),
            Part(
                file_data=FileData(
                    file_uri="gs://bucket/video.mp4",
                    mime_type="video/mp4"
                )
            )
        ]
    )

    # Function calling
    request = GenerateContentRequest(
        contents=[content],
        tools=[Tool(
            function_declarations=[{
                "name": "get_weather",
                "description": "Get weather in location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }]
        )]
    )

    # Safety settings
    request = GenerateContentRequest(
        contents=[content],
        safety_settings=[
            SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            )
        ]
    )
    ```

Response Features:
1. Citations: Automatically links to sources
2. Safety Ratings: Per-response safety scores
3. Multiple Candidates: Alternative responses
4. Token Count: Usage tracking per response
5. Finish Reasons: Detailed completion status
"""

import typing
import pydantic

class InlineData(pydantic.BaseModel):
    mime_type: str
    data: str

class FileData(pydantic.BaseModel):
    file_uri: str
    mime_type: str | None = None

class Part(pydantic.BaseModel):
    text: str | None = None
    inline_data: InlineData | None = None
    file_data: FileData | None = None

class Content(pydantic.BaseModel):
    parts: list[Part]
    role: typing.Literal["user", "model"] | None = None

class SafetySetting(pydantic.BaseModel):
    category: typing.Literal[
        "HARM_CATEGORY_UNSPECIFIED",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_DANGEROUS_CONTENT"
    ]
    threshold: typing.Literal[
        "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
        "BLOCK_LOW_AND_ABOVE",
        "BLOCK_MEDIUM_AND_ABOVE",
        "BLOCK_ONLY_HIGH",
        "BLOCK_NONE"
    ]

class Tool(pydantic.BaseModel):
    function_declarations: list[dict[str, typing.Any]]

class GenerationConfig(pydantic.BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    candidate_count: int | None = None
    max_output_tokens: int | None = None
    stop_sequences: list[str] | None = None

class GenerateContentRequest(pydantic.BaseModel):
    contents: list[Content]
    tools: list[Tool] | None = None
    safety_settings: list[SafetySetting] | None = None
    generation_config: GenerationConfig | None = None

class PromptFeedback(pydantic.BaseModel):
    block_reason: typing.Literal[
        "BLOCK_REASON_UNSPECIFIED",
        "SAFETY",
        "OTHER"
    ] | None = None
    safety_ratings: list[dict[str, str]] | None = None

class CitationSource(pydantic.BaseModel):
    start_index: int
    end_index: int
    uri: str
    license: str

class CitationMetadata(pydantic.BaseModel):
    citation_sources: list[CitationSource]

class Candidate(pydantic.BaseModel):
    content: Content
    finish_reason: typing.Literal[
        "FINISH_REASON_UNSPECIFIED",
        "STOP",
        "MAX_TOKENS",
        "SAFETY",
        "RECITATION",
        "OTHER"
    ] | None = None
    safety_ratings: list[dict[str, str]] | None = None
    citation_metadata: CitationMetadata | None = None
    token_count: int | None = None

class GenerateContentResponse(pydantic.BaseModel):
    candidates: list[Candidate] | None = None
    prompt_feedback: PromptFeedback | None = None

# Streaming models
class StreamGenerateContentResponse(pydantic.BaseModel):
    candidates: list[Candidate] | None = None
    prompt_feedback: PromptFeedback | None = None
    usage_metadata: dict[str, typing.Any] | None = None

# Function calling models
class FunctionResponse(pydantic.BaseModel):
    name: str
    args: dict[str, typing.Any]

class FunctionCall(pydantic.BaseModel):
    name: str
    args: dict[str, typing.Any]
    response: FunctionResponse | None = None