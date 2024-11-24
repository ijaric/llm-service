"""Exceptions for LLM service."""

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    
    def __init__(
        self,
        message: str,
        provider: str,
        code: str | None = None,
        raw_error: Exception | None = None
    ):
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.raw_error = raw_error

class ProviderError(LLMError):
    """Error from LLM provider."""
    pass

class ValidationError(LLMError):
    """Validation error for requests."""
    pass

class UnsupportedOperationError(LLMError):
    """Operation not supported by provider."""
    pass

class ConfigurationError(LLMError):
    """Configuration error."""
    pass

class RateLimitError(ProviderError):
    """Rate limit exceeded."""
    pass

class AuthenticationError(ProviderError):
    """Authentication failed."""
    pass

class QuotaExceededError(ProviderError):
    """Quota exceeded."""
    pass

class InvalidRequestError(ProviderError):
    """Invalid request parameters."""
    pass

class ContextLengthExceededError(ProviderError):
    """Context length exceeded."""
    pass