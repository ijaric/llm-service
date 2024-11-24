"""OpenAI API Models for Chat Completion

This module provides Pydantic models for OpenAI's Chat Completion API.

Key Features:
1. Data Types Support:
   - Text: Regular text messages
   - Images: Base64 or URLs in content array
   - No direct file upload support
   
2. Function Calling:
   - Supports both legacy functions and new tools approach
   - Can force specific function/tool execution
   - Structured function parameters with JSON Schema
   
3. Response Formats:
   - Text (default): Free-form text responses
   - JSON: Structured responses in JSON format (use response_format)
   
4. Response Types:
   - HTTP: Complete response with usage stats
   - Streaming: Progressive response with deltas

Example Usage:
    ```python
    # Image input
    message = Message(
        role="user",
        content=[
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,..."}
            }
        ]
    )

    # Function calling
    request = ChatCompletionRequest(
        model="gpt-4",
        messages=[message],
        tools=[{
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
        }],
        tool_choice="auto"
    )

    # JSON response
    request = ChatCompletionRequest(
        model="gpt-4",
        messages=[message],
        response_format={"type": "json_object"}
    )
    ```
"""

import typing
import pydantic

class FunctionParameter(pydantic.BaseModel):
    type: str
    description: str | None = None
    enum: list[str] | None = None
    items: dict[str, typing.Any] | None = None
    properties: dict[str, 'FunctionParameter'] | None = None
    required: list[str] | None = None

class Function(pydantic.BaseModel):
    name: str
    description: str
    parameters: FunctionParameter

class FunctionCall(pydantic.BaseModel):
    name: str
    arguments: str

class Tool(pydantic.BaseModel):
    type: str = "function"
    function: Function

class ResponseFormat(pydantic.BaseModel):
    type: typing.Literal["text", "json_object"] = "text"

class Message(pydantic.BaseModel):
    role: typing.Literal["system", "user", "assistant", "function", "tool"]
    content: str | list[dict[str, typing.Any]] | None = None
    name: str | None = None
    function_call: FunctionCall | None = None
    tool_calls: list[dict[str, typing.Any]] | None = None

class ChatCompletionRequest(pydantic.BaseModel):
    model: str
    messages: list[Message]
    functions: list[Function] | None = None
    function_call: str | dict[str, str] | None = None
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, str] | None = None
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    n: int | None = 1
    stream: bool | None = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = 0
    frequency_penalty: float | None = 0
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    response_format: ResponseFormat | None = None

class Choice(pydantic.BaseModel):
    index: int
    message: Message
    finish_reason: str | None = None

class Usage(pydantic.BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(pydantic.BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage

# Streaming response models
class DeltaMessage(pydantic.BaseModel):
    role: str | None = None
    content: str | None = None
    function_call: FunctionCall | None = None
    tool_calls: list[dict[str, typing.Any]] | None = None

class StreamChoice(pydantic.BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: str | None = None

class ChatCompletionStreamResponse(pydantic.BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[StreamChoice]

# Content types for multimodal messages
class ContentPart(pydantic.BaseModel):
    type: typing.Literal["text", "image_url", "image_file", "video_file", "audio_file", "pdf_file"]
    text: str | None = None
    image_url: dict[str, str] | None = None
    file_url: dict[str, str] | None = None

class MultiModalMessage(pydantic.BaseModel):
    role: typing.Literal["system", "user", "assistant", "function", "tool"]
    content: list[ContentPart]
    name: str | None = None