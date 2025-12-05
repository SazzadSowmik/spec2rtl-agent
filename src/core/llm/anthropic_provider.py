"""
Anthropic LLM Provider Implementation.

This module provides concrete implementation of BaseLLMProvider for Anthropic's
Claude models, including Claude 3.7 Sonnet, Claude 3.5 Sonnet, and Haiku variants.
"""

import os
from typing import Optional, Dict, Any

from autogen_ext.models.anthropic import AnthropicChatCompletionClient

from .base_provider import BaseLLMProvider, ModelInfo, ModelCapability


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude LLM provider.
    
    Supports:
    - claude-3-7-sonnet-20250219: Latest Claude (smartest)
    - claude-3-5-sonnet-20241022: Previous Sonnet
    - claude-3-5-haiku-20241022: Fast, cheap variant
    - claude-3-opus-20240229: Older, most capable (legacy)
    
    Key Features:
    - Long context: Up to 200k tokens
    - Vision support: All models support images
    - Tool calling: Native function calling support
    - Streaming: Full streaming support
    
    Attributes:
        api_key: Anthropic API key (from env or explicit)
        base_url: Optional custom API endpoint
    
    Example:
        >>> provider = AnthropicProvider(
        ...     model_name="claude-3-7-sonnet-20250219",
        ...     max_tokens=4096,
        ...     temperature=0.3
        ... )
        >>> client = provider.create_model_client()
    """
    
    # Model pricing (as of December 2024, per 1M tokens)
    MODEL_PRICING = {
        # Claude 3.7 (latest)
        "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
        
        # Claude 3.5 family
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        
        # Claude 3 family (legacy)
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }
    
    # Context windows (tokens)
    CONTEXT_WINDOWS = {
        "claude-3-7-sonnet-20250219": 200000,
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-sonnet-20240620": 200000,
        "claude-3-5-haiku-20241022": 200000,
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
    }
    
    def __init__(
        self,
        model_name: str = "claude-3-7-sonnet-20250219",
        temperature: Optional[float] = 0.3,
        max_tokens: Optional[int] = 4096,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Anthropic provider.
        
        Args:
            model_name: Claude model name (default: claude-3-7-sonnet-20250219)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate (required by Anthropic)
            api_key: Anthropic API key (default: from ANTHROPIC_API_KEY env var)
            base_url: Custom API endpoint (for proxies)
            **kwargs: Additional arguments for AnthropicChatCompletionClient
        
        Raises:
            ValueError: If model is invalid or API key is missing
        
        Note:
            Anthropic requires max_tokens to be explicitly set (no default).
            If not provided, defaults to 4096.
        """
        # Store Anthropic-specific parameters
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = base_url
        
        # Validate API key
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter."
            )
        
        # Anthropic requires max_tokens (no default in API)
        if max_tokens is None:
            max_tokens = 4096
            print(f"Warning: max_tokens not specified. Using default: {max_tokens}")
        
        # Validate model exists
        if not self._is_known_model(model_name):
            print(f"Warning: Unknown model '{model_name}'. Pricing may be inaccurate.")
        
        # Initialize base class
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "anthropic"
    
    def create_model_client(self) -> AnthropicChatCompletionClient:
        """
        Create Anthropic ChatCompletionClient for AutoGen.
        
        Returns:
            Configured AnthropicChatCompletionClient instance
        
        Raises:
            ImportError: If autogen_ext.models.anthropic is not installed
        """
        # Build client configuration
        client_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "api_key": self.api_key,
            "max_tokens": self.max_tokens,  # Required by Anthropic
        }
        
        # Add optional parameters
        if self.temperature is not None:
            client_kwargs["temperature"] = self.temperature
        
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        # Add extra parameters from kwargs
        client_kwargs.update(self.extra_params)
        
        return AnthropicChatCompletionClient(**client_kwargs)
    
    def get_model_info(self) -> ModelInfo:
        """
        Get model capabilities and pricing information.
        
        Returns:
            ModelInfo with capabilities and cost data
        """
        # Get pricing
        pricing = self.MODEL_PRICING.get(
            self.model_name,
            {"input": 0.0, "output": 0.0}  # Unknown models
        )
        
        # Get context window
        context_window = self.CONTEXT_WINDOWS.get(self.model_name, 200000)
        
        # All Claude 3+ models support vision and tools
        is_claude_3_plus = self.model_name.startswith("claude-3")
        
        return ModelInfo(
            name=self.model_name,
            provider=self.provider_name,
            context_window=context_window,
            supports_streaming=True,
            supports_function_calling=is_claude_3_plus,
            supports_vision=is_claude_3_plus,
            is_reasoning_model=False,  # Claude doesn't have reasoning models
            cost_per_1k_input_tokens=pricing["input"] / 1000,
            cost_per_1k_output_tokens=pricing["output"] / 1000,
        )
    
    def _is_known_model(self, model_name: str) -> bool:
        """Check if model is in known models list."""
        return model_name in self.MODEL_PRICING
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AnthropicProvider("
            f"model={self.model_name}, "
            f"total_cost=${self._total_usage.estimated_cost:.4f})"
        )