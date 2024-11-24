"""LLM provider interface definition."""

import typing

import models.base as base

class LLMProvider(typing.Protocol):
    """Protocol defining interface for LLM providers."""
    
    @property
    def name(self) -> str:
        """Provider name (e.g. 'openai', 'anthropic', 'gemini')."""
        ...
    
    @property
    def supported_content_types(self) -> set[base.ContentType]:
        """Set of supported content types."""
        ...
    
    @property
    def supports_functions(self) -> bool:
        """Whether provider supports function calling."""
        ...
    
    @property
    def supports_json_response(self) -> bool:
        """Whether provider supports native JSON responses."""
        ...

    async def validate_request(
        self,
        request: base.Request
    ) -> list[str]:
        """Validate request and return list of validation errors.
        
        Empty list means request is valid.
        """
        ...

    async def convert_request(
        self,
        request: base.Request
    ) -> typing.Any:
        """Convert universal request to provider-specific format."""
        ...

    async def convert_response(
        self,
        raw_response: typing.Any,
        request: base.Request
    ) -> base.Response:
        """Convert provider-specific response to universal format."""
        ...

    async def convert_stream_response(
        self,
        raw_chunk: typing.Any,
        request: base.Request
    ) -> base.StreamResponse:
        """Convert provider-specific stream chunk to universal format."""
        ...

    async def convert_error(
        self,
        error: Exception
    ) -> base.LLMError:
        """Convert provider-specific error to universal format."""
        ...