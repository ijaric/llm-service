"""Models package for LLM interactions."""

from models.base import (
    # Common
    ContentType, Role, MediaSource, Content, 
    FunctionParameter, Function, FunctionCall,
    Usage, ResponseMetadata, LLMError,
    # Chat
    ChatRequest, ChatResponse, ChatStreamResponse,
    # Embeddings
    EmbeddingRequest, EmbeddingResponse, EmbeddingData,
    # Speech
    Voice, AudioFormat, SpeechRequest, SpeechResponse,
)

from models.capabilities import (
    ChatProvider,
    EmbeddingProvider,
    SpeechProvider,
)