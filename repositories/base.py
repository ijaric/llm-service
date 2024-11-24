"""Base repository implementations."""

import logging
import typing

import lib.app.settings as settings
import models.base as base
import models.capabilities as capabilities
import utils.fetch as fetch
import utils.logging as logging_utils
from exceptions import (
    UnsupportedOperationError,
    ValidationError,
    ConfigurationError
)

logger = logging.getLogger(__name__)

class BaseLLMRepository:
    """Base repository with common functionality."""
    
    base_url: str
    provider_name: str
    
    def __init__(self, api_key: str | None = None):
        """Initialize repository."""
        self.api_key = api_key or self._get_api_key()
        
    def _get_api_key(self) -> str:
        """Get API key from settings."""
        raise NotImplementedError(
            f"API key retrieval not implemented for {self.provider_name}"
        )
    
    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        raise NotImplementedError(
            f"Headers not implemented for {self.provider_name}"
        )
        
    def _validate_configuration(self) -> None:
        """Validate repository configuration."""
        if not self.api_key:
            raise ConfigurationError(
                f"API key not configured for {self.provider_name}",
                provider=self.provider_name
            )

class ChatRepository(BaseLLMRepository):
    """Repository for chat completion capabilities."""
    
    chat_response_model: type[typing.Any]
    
    def __init__(self, provider: capabilities.ChatProvider):
        self.provider = provider
        
    async def complete(
        self,
        request: base.ChatRequest
    ) -> base.ChatResponse:
        """Send chat completion request."""
        errors = await self.provider.validate_chat_request(request)
        if errors:
            raise ValueError(errors)
            
        provider_request = await self.provider.convert_chat_request(request)
        
        try:
            raw_response = await fetch.make_request(
                url=f"{self.base_url}/v1/chat/completions",
                method="POST",
                headers=self._get_headers(),
                body=provider_request,
                model=self.chat_response_model
            )
            return await self.provider.convert_chat_response(raw_response, request)
        except Exception as e:
            raise await self.provider.convert_error(e)

class EmbeddingRepository(BaseLLMRepository):
    """Repository for embedding capabilities."""
    
    embedding_response_model: type[typing.Any]
    
    def __init__(self, provider: capabilities.EmbeddingProvider):
        self.provider = provider
        
    async def embed(
        self,
        request: base.EmbeddingRequest
    ) -> base.EmbeddingResponse:
        """Generate embeddings for input."""
        errors = await self.provider.validate_embedding_request(request)
        if errors:
            raise ValueError(errors)
            
        provider_request = await self.provider.convert_embedding_request(request)
        
        try:
            raw_response = await fetch.make_request(
                url=f"{self.base_url}/v1/embeddings",
                method="POST",
                headers=self._get_headers(),
                body=provider_request,
                model=self.embedding_response_model
            )
            return await self.provider.convert_embedding_response(raw_response, request)
        except Exception as e:
            raise await self.provider.convert_error(e)

class SpeechRepository(BaseLLMRepository):
    """Repository for speech synthesis capabilities."""
    
    speech_response_model: type[typing.Any]
    
    def __init__(self, provider: capabilities.SpeechProvider):
        self.provider = provider
        
    async def synthesize(
        self,
        request: base.SpeechRequest
    ) -> base.SpeechResponse:
        """Generate speech from text."""
        errors = await self.provider.validate_speech_request(request)
        if errors:
            raise ValueError(errors)
            
        provider_request = await self.provider.convert_speech_request(request)
        
        try:
            raw_response = await fetch.make_request(
                url=f"{self.base_url}/v1/audio/speech",
                method="POST",
                headers=self._get_headers(),
                body=provider_request,
                model=self.speech_response_model
            )
            return await self.provider.convert_speech_response(raw_response, request)
        except Exception as e:
            raise await self.provider.convert_error(e)