"""LLM service implementation."""

import models.base as base
from repositories.base import ChatRepository, EmbeddingRepository, SpeechRepository

class LLMService:
    """Service for LLM interactions."""
    
    def __init__(
        self,
        chat_repo: ChatRepository | None = None,
        embedding_repo: EmbeddingRepository | None = None,
        speech_repo: SpeechRepository | None = None
    ):
        """Initialize service with optional repositories."""
        self.chat_repo = chat_repo
        self.embedding_repo = embedding_repo
        self.speech_repo = speech_repo
    
    async def complete(
        self,
        request: base.ChatRequest
    ) -> base.ChatResponse:
        """Send chat completion request."""
        if not self.chat_repo:
            raise NotImplementedError("Chat completion not supported")
        return await self.chat_repo.complete(request)
    
    async def embed(
        self,
        request: base.EmbeddingRequest
    ) -> base.EmbeddingResponse:
        """Generate embeddings for input."""
        if not self.embedding_repo:
            raise NotImplementedError("Embeddings not supported")
        return await self.embedding_repo.embed(request)
    
    async def synthesize(
        self,
        request: base.SpeechRequest
    ) -> base.SpeechResponse:
        """Generate speech from text."""
        if not self.speech_repo:
            raise NotImplementedError("Speech synthesis not supported")
        return await self.speech_repo.synthesize(request)