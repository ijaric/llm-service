"""Google Gemini repository implementations."""

import lib.app.settings as settings
import models.gemini as gemini_models
from repositories.base import ChatRepository, EmbeddingRepository
from exceptions import UnsupportedOperationError

class GeminiBaseMixin:
    """Mixin with common Gemini functionality."""
    
    provider_name = "gemini"
    base_url = "https://generativelanguage.googleapis.com"
    
    def __init__(self, api_key: str | None = None, version: str = "v1"):
        super().__init__(api_key)
        self.version = version or settings.llms.gemini_version or "v1"
    
    def _get_api_key(self) -> str:
        """Get Gemini API key from settings."""
        if not settings.llms.gemini_api_key:
            raise ConfigurationError(
                "Gemini API key not found in settings",
                provider=self.provider_name
            )
        return settings.llms.gemini_api_key
    
    def _get_headers(self) -> dict[str, str]:
        """Get Gemini API headers."""
        return {
            "Content-Type": "application/json"
        }

class GeminiChatRepository(GeminiBaseMixin, ChatRepository):
    """Gemini chat completion repository."""
    
    chat_response_model = gemini_models.GenerateContentResponse
    
    @logging_utils.log_operation(
        provider="gemini",
        operation="chat_completion"
    )
    async def complete(
        self,
        request: base.ChatRequest
    ) -> base.ChatResponse:
        """Override to handle API key in URL."""
        self._validate_configuration()
        
        if not isinstance(self.provider, capabilities.ChatProvider):
            raise UnsupportedOperationError(
                "Chat completion not supported by Gemini",
                provider=self.provider_name
            )
        
        errors = await self.provider.validate_chat_request(request)
        if errors:
            raise ValidationError(
                f"Invalid request: {', '.join(errors)}",
                provider=self.provider_name
            )
            
        provider_request = await self.provider.convert_chat_request(request)
        
        try:
            raw_response = await fetch.make_request(
                url=f"{self.base_url}/{self.version}/models/{request.model}:generateContent",
                method="POST",
                headers=self._get_headers(),
                params={"key": self.api_key},
                body=provider_request,
                model=self.chat_response_model
            )
            return await self.provider.convert_chat_response(raw_response, request)
        except Exception as e:
            raise await self.provider.convert_error(e)

class GeminiEmbeddingRepository(GeminiBaseMixin, EmbeddingRepository):
    """Gemini embeddings repository."""
    
    embedding_response_model = gemini_models.EmbeddingResponse
    
    @logging_utils.log_operation(
        provider="gemini",
        operation="embeddings"
    )
    async def embed(
        self,
        request: base.EmbeddingRequest
    ) -> base.EmbeddingResponse:
        """Override to handle API key in URL."""
        self._validate_configuration()
        
        if not isinstance(self.provider, capabilities.EmbeddingProvider):
            raise UnsupportedOperationError(
                "Embeddings not supported by Gemini",
                provider=self.provider_name
            )
        
        errors = await self.provider.validate_embedding_request(request)
        if errors:
            raise ValidationError(
                f"Invalid request: {', '.join(errors)}",
                provider=self.provider_name
            )
            
        provider_request = await self.provider.convert_embedding_request(request)
        
        try:
            raw_response = await fetch.make_request(
                url=f"{self.base_url}/{self.version}/models/{request.model}:embedContent",
                method="POST",
                headers=self._get_headers(),
                params={"key": self.api_key},
                body=provider_request,
                model=self.embedding_response_model
            )
            return await self.provider.convert_embedding_response(raw_response, request)
        except Exception as e:
            raise await self.provider.convert_error(e)