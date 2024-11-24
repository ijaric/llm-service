"""LLM provider capabilities and interfaces."""

import typing

import models.base as base

class ChatProvider(typing.Protocol):
    """Protocol for chat completion capabilities."""
    
    @property
    def name(self) -> str:
        """Provider name."""
        ...
    
    @property
    def supported_content_types(self) -> set[base.ContentType]:
        """Supported content types for chat."""
        ...
    
    @property
    def supports_functions(self) -> bool:
        """Whether provider supports function calling."""
        ...
    
    @property
    def supports_json_response(self) -> bool:
        """Whether provider supports native JSON responses."""
        ...

    async def validate_chat_request(
        self,
        request: base.ChatRequest
    ) -> list[str]:
        """Validate chat request."""
        ...

    async def convert_chat_request(
        self,
        request: base.ChatRequest
    ) -> typing.Any:
        """Convert universal chat request to provider format."""
        ...

    async def convert_chat_response(
        self,
        raw_response: typing.Any,
        request: base.ChatRequest
    ) -> base.ChatResponse:
        """Convert provider chat response to universal format."""
        ...

    async def convert_chat_stream(
        self,
        raw_chunk: typing.Any,
        request: base.ChatRequest
    ) -> base.ChatStreamResponse:
        """Convert provider stream chunk to universal format."""
        ...

class EmbeddingProvider(typing.Protocol):
    """Protocol for embedding capabilities."""
    
    @property
    def name(self) -> str:
        """Provider name."""
        ...
    
    @property
    def supported_content_types(self) -> set[base.ContentType]:
        """Supported content types for embeddings."""
        ...
    
    @property
    def embedding_dimensions(self) -> int:
        """Number of dimensions in the embedding vector."""
        ...

    async def validate_embedding_request(
        self,
        request: base.EmbeddingRequest
    ) -> list[str]:
        """Validate embedding request."""
        ...

    async def convert_embedding_request(
        self,
        request: base.EmbeddingRequest
    ) -> typing.Any:
        """Convert universal embedding request to provider format."""
        ...

    async def convert_embedding_response(
        self,
        raw_response: typing.Any,
        request: base.EmbeddingRequest
    ) -> base.EmbeddingResponse:
        """Convert provider embedding response to universal format."""
        ...

class SpeechProvider(typing.Protocol):
    """Protocol for speech capabilities."""
    
    @property
    def name(self) -> str:
        """Provider name."""
        ...
    
    @property
    def supported_voices(self) -> list[str]:
        """List of supported voices."""
        ...
    
    @property
    def supported_audio_formats(self) -> set[str]:
        """Supported audio formats."""
        ...

    async def validate_speech_request(
        self,
        request: base.SpeechRequest
    ) -> list[str]:
        """Validate speech request."""
        ...

    async def convert_speech_request(
        self,
        request: base.SpeechRequest
    ) -> typing.Any:
        """Convert universal speech request to provider format."""
        ...

    async def convert_speech_response(
        self,
        raw_response: typing.Any,
        request: base.SpeechRequest
    ) -> base.SpeechResponse:
        """Convert provider speech response to universal format."""
        ...