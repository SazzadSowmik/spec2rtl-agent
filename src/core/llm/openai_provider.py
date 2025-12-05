"""
OpenAI LLM Provider Implementation.

This module provides concrete implementation of BaseLLMProvider for OpenAI models,
including GPT-4o, GPT-4o-mini, and o1 reasoning models.
"""

import os
from typing import Optional, Dict, Any

from autogen_ext.models.openai import OpenAIChatCompletionClient

from .base_provider import BaseLLMProvider, ModelInfo, ModelCapability


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider supporting GPT-4o and o1 model families.
    
    Supports:
    - gpt-4o: Latest GPT-4 Omni model (multimodal, 128k context)
    - gpt-4o-mini: Faster, cheaper variant
    - o1-preview: Reasoning model (no temperature control)
    - o1-mini: Smaller reasoning model
    - gpt-4-turbo: Previous generation
    
    Attributes:
        api_key: OpenAI API key (from env or explicit)
        base_url: Optional custom API endpoint
        organization: Optional organization ID
    
    Example:
        >>> provider = OpenAIProvider(model_name="gpt-4o", temperature=0.3)
        >>> client = provider.create_model_client()
        >>> usage = provider.get_total_usage()
    """
    
    # Model pricing (as of December 2024, per 1M tokens)
    MODEL_PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
        "o1-preview": {"input": 15.00, "output": 60.00},
        "o1-preview-2024-09-12": {"input": 15.00, "output": 60.00},
        "o1-mini": {"input": 3.00, "output": 12.00},
        "o1-mini-2024-09-12": {"input": 3.00, "output": 12.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4-turbo-2024-04-09": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
    }
    
    # Reasoning models (different parameter support)
    REASONING_MODELS = {"o1-preview", "o1-mini", "o1-preview-2024-09-12", "o1-mini-2024-09-12"}
    
    # Context windows
    CONTEXT_WINDOWS = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "o1-preview": 128000,
        "o1-mini": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
    }
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: Optional[float] = 0.3,
        max_tokens: Optional[int] = 4096,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            model_name: OpenAI model name (default: gpt-4o)
            temperature: Sampling temperature, ignored for o1 models
            max_tokens: Maximum tokens (converted to max_completion_tokens for o1)
            api_key: OpenAI API key (default: from OPENAI_API_KEY env var)
            base_url: Custom API endpoint (for proxies/compatible APIs)
            organization: OpenAI organization ID
            **kwargs: Additional arguments for OpenAIChatCompletionClient
        
        Raises:
            ValueError: If model is invalid or API key is missing
        """
        # Store OpenAI-specific parameters
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.organization = organization
        
        # Validate API key
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
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
        return "openai"
    
    def create_model_client(self) -> OpenAIChatCompletionClient:
        """
        Create OpenAI ChatCompletionClient for AutoGen.
        
        Returns:
            Configured OpenAIChatCompletionClient instance
        
        Raises:
            ImportError: If autogen_ext.models.openai is not installed
        """
        # Build client configuration
        client_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "api_key": self.api_key,
        }
        
        # Add optional parameters
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        if self.organization:
            client_kwargs["organization"] = self.organization
        
        # Handle model-specific parameters
        is_reasoning = self._is_reasoning_model()
        
        if is_reasoning:
            # o1 models use max_completion_tokens instead of max_tokens
            if self.max_tokens:
                client_kwargs["max_completion_tokens"] = self.max_tokens
            # o1 models don't support temperature, top_p, etc.
            # Don't add temperature for reasoning models
        else:
            # Traditional models support temperature and max_tokens
            if self.temperature is not None:
                client_kwargs["temperature"] = self.temperature
            if self.max_tokens:
                client_kwargs["max_tokens"] = self.max_tokens
        
        # Add extra parameters from kwargs
        client_kwargs.update(self.extra_params)
        
        return OpenAIChatCompletionClient(**client_kwargs)
    
    def get_model_info(self) -> ModelInfo:
        """
        Get model capabilities and pricing information.
        
        Returns:
            ModelInfo with capabilities and cost data
        """
        is_reasoning = self._is_reasoning_model()
        
        # Get pricing (use base model name for versioned models)
        base_model = self._get_base_model_name()
        pricing = self.MODEL_PRICING.get(
            base_model,
            {"input": 0.0, "output": 0.0}  # Unknown models
        )
        
        # Get context window
        context_window = self.CONTEXT_WINDOWS.get(base_model, 8192)
        
        return ModelInfo(
            name=self.model_name,
            provider=self.provider_name,
            context_window=context_window,
            supports_streaming=True,
            supports_function_calling=not is_reasoning,  # o1 doesn't support tools yet
            supports_vision="gpt-4o" in self.model_name or "gpt-4-turbo" in self.model_name,
            is_reasoning_model=is_reasoning,
            cost_per_1k_input_tokens=pricing["input"] / 1000,  # Convert to per 1k
            cost_per_1k_output_tokens=pricing["output"] / 1000,
        )
    
    def _is_reasoning_model(self) -> bool:
        """Check if current model is a reasoning model (o1 series)."""
        return any(rm in self.model_name for rm in self.REASONING_MODELS)
    
    def _is_known_model(self, model_name: str) -> bool:
        """Check if model is in known models list."""
        base_name = self._get_base_model_name(model_name)
        return base_name in self.MODEL_PRICING
    
    def _get_base_model_name(self, model_name: Optional[str] = None) -> str:
        """
        Extract base model name from versioned model string.
        
        Example: "gpt-4o-2024-11-20" -> "gpt-4o"
        """
        name = model_name or self.model_name
        
        # Handle versioned models
        if "-2024-" in name or "-2023-" in name:
            # Extract base name before date
            parts = name.split("-")
            # Find where date starts (year format)
            for i, part in enumerate(parts):
                if part.isdigit() and len(part) == 4:
                    return "-".join(parts[:i])
        
        return name
    
    def __repr__(self) -> str:
        """String representation."""
        model_type = "reasoning" if self._is_reasoning_model() else "standard"
        return (
            f"OpenAIProvider("
            f"model={self.model_name}, "
            f"type={model_type}, "
            f"total_cost=${self._total_usage.estimated_cost:.4f})"
        )