"""Anthropic repository implementations."""

import lib.app.settings as settings
import models.anthropic as anthropic_models
from repositories.base import ChatRepository
from exceptions import UnsupportedOperationError

class AnthropicBaseMixin:
    """Mixin with common Anthropic functionality."""
    
    provider_name = "anthropic"
    base_url = "https://api.anthropic.com"
    
    def _get_api_key(self) -> str:
        """Get Anthropic API key from settings."""
        if not settings.llms.anthropic_api_key:
            raise ConfigurationError(
                "Anthropic API key not found in settings",
                provider=self.provider_name
            )
        return settings.llms.anthropic_api_key
    
    def _get_headers(self) -> dict[str, str]:
        """Get Anthropic API headers."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": settings.llms.anthropic_version or "2023-06-01",
            "Content-Type": "application/json"
        }

class AnthropicChatRepository(AnthropicBaseMixin, ChatRepository):
    """Anthropic chat completion repository."""
    
    chat_response_model = anthropic_models.MessageResponse