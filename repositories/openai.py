"""OpenAI repository implementations."""

import lib.app.settings as settings
import models.openai as openai_models
from repositories.base import ChatRepository, EmbeddingRepository

class OpenAIBaseMixin:
    """Mixin with common OpenAI functionality."""
    
    provider_name = "openai"
    base_url = "https://api.openai.com"
    
    def _get_api_key(self) -> str:
        """Get OpenAI API key from settings."""
        if not settings.llms.openai_api_key:
            raise ConfigurationError(
                "OpenAI API key not found in settings",
                provider=self.provider_name
            )
        return settings.llms.openai_api_key
    
    def _get_headers(self) -> dict[str, str]:
        """Get OpenAI API headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "OpenAI-Organization": settings.llms.openai_org_id or ""
        }

class OpenAIChatRepository(OpenAIBaseMixin, ChatRepository):
    """OpenAI chat completion repository."""
    
    chat_response_model = openai_models.ChatCompletionResponse

class OpenAIEmbeddingRepository(OpenAIBaseMixin, EmbeddingRepository):
    """OpenAI embeddings repository."""
    
    embedding_response_model = openai_models.EmbeddingResponse