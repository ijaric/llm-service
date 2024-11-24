"""Universal interface for LLM interactions.

This module provides base models for unified LLM request/response handling,
allowing easy switching between different LLM providers while maintaining
a consistent interface.
"""

import enum
import typing
import pydantic

class ContentType(str, enum.Enum):
    """Supported content types for LLM interactions."""
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    VIDEO = "video"
    AUDIO = "audio"

class Role(str, enum.Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"

class MediaSource(pydantic.BaseModel):
    """Universal media content representation."""
    type: ContentType
    mime_type: str
    # One of the following should be provided
    data: str | None = None  # base64 encoded content
    url: str | None = None   # HTTP/HTTPS URL
    path: str | None = None  # File system path

    model_config = {
        "extra": "forbid"
    }

class Content(pydantic.BaseModel):
    """Content part of the message."""
    type: ContentType
    text: str | None = None
    media: MediaSource | None = None

    model_config = {
        "extra": "forbid"
    }

class FunctionParameter(pydantic.BaseModel):
    """Function parameter definition."""
    type: str
    description: str | None = None
    enum: list[str] | None = None
    items: dict[str, typing.Any] | None = None
    properties: dict[str, 'FunctionParameter'] | None = None
    required: list[str] | None = None

    model_config = {
        "extra": "forbid"
    }

class Function(pydantic.BaseModel):
    """Function definition."""
    name: str
    description: str
    parameters: FunctionParameter

    model_config = {
        "extra": "forbid"
    }

class FunctionCall(pydantic.BaseModel):
    """Function call details."""
    name: str
    arguments: dict[str, typing.Any]
    response: dict[str, typing.Any] | None = None

    model_config = {
        "extra": "forbid"
    }

class Message(pydantic.BaseModel):
    """Single message in conversation."""
    role: Role
    content: list[Content]
    name: str | None = None
    function_call: FunctionCall | None = None

    model_config = {
        "extra": "forbid"
    }

class GenerationConfig(pydantic.BaseModel):
    """Common generation parameters."""
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    stream: bool = False
    functions: list[Function] | None = None
    force_function: str | None = None  # Force specific function call
    response_format: typing.Literal["text", "json"] = "text"

    model_config = {
        "extra": "forbid"
    }

# Chat Completion Models
class ChatRequest(pydantic.BaseModel):
    """Universal chat completion request."""
    model: str
    messages: list[Message]
    config: GenerationConfig = GenerationConfig()

    model_config = {
        "extra": "forbid"
    }

class ChatResponse(pydantic.BaseModel):
    """Universal chat completion response."""
    id: str
    content: list[Content]
    function_calls: list[FunctionCall] | None = None
    metadata: ResponseMetadata

    model_config = {
        "extra": "forbid"
    }

class ChatStreamResponse(pydantic.BaseModel):
    """Universal chat streaming response chunk."""
    id: str
    delta: Content | FunctionCall
    metadata: ResponseMetadata | None = None
    done: bool = False

    model_config = {
        "extra": "forbid"
    }

# Embedding Models
class EmbeddingRequest(pydantic.BaseModel):
    """Universal embedding request."""
    model: str
    input: list[str | Content]
    encoding_format: typing.Literal["float", "base64"] = "float"

    model_config = {
        "extra": "forbid"
    }

class EmbeddingData(pydantic.BaseModel):
    """Single embedding result."""
    index: int
    embedding: list[float]
    object: typing.Literal["embedding"]

class EmbeddingResponse(pydantic.BaseModel):
    """Universal embedding response."""
    id: str
    data: list[EmbeddingData]
    model: str
    object: typing.Literal["list"]
    usage: Usage

    model_config = {
        "extra": "forbid"
    }

# Speech Models
class Voice(str, enum.Enum):
    """Common voice types across providers."""
    MALE_1 = "male-1"
    MALE_2 = "male-2"
    FEMALE_1 = "female-1"
    FEMALE_2 = "female-2"
    NEUTRAL = "neutral"

class AudioFormat(str, enum.Enum):
    """Supported audio formats."""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    FLAC = "flac"

class SpeechRequest(pydantic.BaseModel):
    """Universal speech synthesis request."""
    model: str
    input: str
    voice: Voice
    format: AudioFormat = AudioFormat.MP3
    speed: float = 1.0
    pitch: float | None = None
    volume: float | None = None

    model_config = {
        "extra": "forbid"
    }

class SpeechResponse(pydantic.BaseModel):
    """Universal speech synthesis response."""
    id: str
    audio: bytes
    duration: float
    format: AudioFormat
    metadata: ResponseMetadata

    model_config = {
        "extra": "forbid"
    }

# Common Models
class Usage(pydantic.BaseModel):
    """Token/resource usage statistics."""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    # For audio
    audio_duration: float | None = None
    # For embeddings
    total_dimensions: int | None = None

    model_config = {
        "extra": "forbid"
    }

class ResponseMetadata(pydantic.BaseModel):
    """Additional response information."""
    model: str
    usage: Usage | None = None
    finish_reason: str | None = None
    # Provider-specific metadata
    raw_response: dict[str, typing.Any] | None = None

    model_config = {
        "extra": "forbid"
    }

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    def __init__(
        self,
        message: str,
        provider: str,
        code: str | None = None,
        raw_error: typing.Any | None = None
    ):
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.raw_error = raw_error