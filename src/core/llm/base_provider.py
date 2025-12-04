"""
Abstract base class for LLM providers.

This module defines the interface that all LLM provider implementations
must follow, ensuring consistency across OpenAI, Anthropic, and local models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING

# Conditional imports for model clients
try:
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    if TYPE_CHECKING:
        from autogen_ext.models.openai import OpenAIChatCompletionClient

try:
    from autogen_ext.models.anthropic import AnthropicChatCompletionClient
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    if TYPE_CHECKING:
        from autogen_ext.models.anthropic import AnthropicChatCompletionClient

# Type alias for model clients
if TYPE_CHECKING:
    ChatCompletionClient = Union[OpenAIChatCompletionClient, AnthropicChatCompletionClient]
else:
    ChatCompletionClient = Any


class ModelCapability(Enum):
    """Model capabilities that may vary across providers."""
    TEMPERATURE_CONTROL = "temperature"
    MAX_TOKENS_CONTROL = "max_tokens"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"
    VISION = "vision"
    JSON_MODE = "json_mode"


@dataclass
class ModelInfo:
    """Information about a specific model."""
    name: str
    provider: str
    context_window: int
    supports_streaming: bool = True
    supports_function_calling: bool = True
    supports_vision: bool = False
    is_reasoning_model: bool = False  # For o1 models
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    
    def __post_init__(self):
        """Validate model info."""
        if self.context_window <= 0:
            raise ValueError("context_window must be positive")


@dataclass
class UsageMetrics:
    """Token usage and cost tracking."""
    timestamp: datetime = field(default_factory=datetime.now)
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    
    def __add__(self, other: 'UsageMetrics') -> 'UsageMetrics':
        """Combine usage metrics."""
        if not isinstance(other, UsageMetrics):
            return NotImplemented
        
        return UsageMetrics(
            timestamp=self.timestamp,
            provider=self.provider,
            model=self.model,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost=self.estimated_cost + other.estimated_cost,
        )


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All provider implementations (OpenAI, Anthropic, local) must inherit
    from this class and implement its abstract methods.
    
    Attributes:
        model_name: Name of the model (e.g., "gpt-4o", "claude-3-7-sonnet")
        temperature: Sampling temperature (if supported)
        max_tokens: Maximum tokens to generate (if supported)
        _model_info: Information about the model's capabilities
        _total_usage: Cumulative token usage and costs
    """
    
    def __init__(
        self,
        model_name: str,
        temperature: Optional[float] = 0.3,
        max_tokens: Optional[int] = 4096,
        **kwargs
    ):
        """
        Initialize the provider.
        
        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature (0.0-2.0), ignored for reasoning models
            max_tokens: Maximum tokens to generate, converted to max_completion_tokens for o1
            **kwargs: Additional provider-specific parameters
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs
        
        self._model_info: Optional[ModelInfo] = None
        self._total_usage = UsageMetrics()
        
        # Validate on initialization
        self._validate_config()
    
    @abstractmethod
    def create_model_client(self) -> ChatCompletionClient:
        """
        Create and return an AutoGen ChatCompletionClient.
        
        Returns:
            Configured ChatCompletionClient for use with AutoGen agents
            
        Raises:
            ValueError: If configuration is invalid
            EnvironmentError: If API keys are missing
            ImportError: If required provider extension is not installed
        """
        pass

    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the model's capabilities and pricing.
        
        Returns:
            ModelInfo object with model capabilities and cost information
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Return the provider name.
        
        Returns:
            Provider name (e.g., "openai", "anthropic", "local")
        """
        pass
    
    def _validate_config(self) -> None:
        """
        Validate provider configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        
        if self.temperature is not None:
            if not 0.0 <= self.temperature <= 2.0:
                raise ValueError("temperature must be between 0.0 and 2.0")
        
        if self.max_tokens is not None:
            if self.max_tokens <= 0:
                raise ValueError("max_tokens must be positive")
    
    def supports_capability(self, capability: ModelCapability) -> bool:
        """
        Check if the model supports a specific capability.
        
        Args:
            capability: ModelCapability to check
            
        Returns:
            True if capability is supported, False otherwise
        """
        model_info = self.get_model_info()
        
        capability_map = {
            ModelCapability.TEMPERATURE_CONTROL: not model_info.is_reasoning_model,
            ModelCapability.MAX_TOKENS_CONTROL: True,
            ModelCapability.FUNCTION_CALLING: model_info.supports_function_calling,
            ModelCapability.STREAMING: model_info.supports_streaming,
            ModelCapability.VISION: model_info.supports_vision,
            ModelCapability.JSON_MODE: not model_info.is_reasoning_model,
        }
        
        return capability_map.get(capability, False)
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate estimated cost for token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        model_info = self.get_model_info()
        
        input_cost = (input_tokens / 1000) * model_info.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * model_info.cost_per_1k_output_tokens
        
        return input_cost + output_cost
    
    def track_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UsageMetrics:
        """
        Track token usage and calculate costs.
        
        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
            metadata: Optional metadata about the request
            
        Returns:
            UsageMetrics object with usage and cost information
        """
        total_tokens = input_tokens + output_tokens
        estimated_cost = self.calculate_cost(input_tokens, output_tokens)
        
        usage = UsageMetrics(
            timestamp=datetime.now(),
            provider=self.provider_name,
            model=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
        )
        
        # Accumulate total usage
        self._total_usage = self._total_usage + usage
        
        return usage
    
    def get_total_usage(self) -> UsageMetrics:
        """
        Get cumulative token usage and costs.
        
        Returns:
            UsageMetrics object with total accumulated usage
        """
        return self._total_usage
    
    def reset_usage(self) -> None:
        """Reset usage tracking counters."""
        self._total_usage = UsageMetrics(
            provider=self.provider_name,
            model=self.model_name
        )
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get provider configuration as dictionary.
        
        Returns:
            Dictionary with all configuration parameters
        """
        config = {
            "provider": self.provider_name,
            "model": self.model_name,
        }
        
        # Only include temperature/max_tokens if supported
        if self.supports_capability(ModelCapability.TEMPERATURE_CONTROL):
            config["temperature"] = self.temperature
        
        if self.max_tokens is not None:
            # Use appropriate key based on model type
            model_info = self.get_model_info()
            if model_info.is_reasoning_model:
                config["max_completion_tokens"] = self.max_tokens
            else:
                config["max_tokens"] = self.max_tokens
        
        # Add extra params
        config.update(self.extra_params)
        
        return config
    
    def __repr__(self) -> str:
        """String representation of the provider."""
        return (
            f"{self.__class__.__name__}("
            f"provider={self.provider_name}, "
            f"model={self.model_name}, "
            f"total_cost=${self._total_usage.estimated_cost:.4f})"
        )