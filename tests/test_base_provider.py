"""Test the base provider abstraction."""

import pytest
from src.core.llm.base_provider import (
    BaseLLMProvider,
    ModelInfo,
    ModelCapability,
    UsageMetrics,
)


class MockProvider(BaseLLMProvider):
    """Mock provider for testing."""
    
    @property
    def provider_name(self) -> str:
        return "mock"
    
    def create_model_client(self):
        return None
    
    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.model_name,
            provider="mock",
            context_window=8192,
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=0.03,
        )


def test_provider_initialization():
    """Test provider can be initialized with valid config."""
    provider = MockProvider(model_name="test-model")
    assert provider.model_name == "test-model"
    assert provider.temperature == 0.3
    assert provider.max_tokens == 4096


def test_invalid_temperature():
    """Test invalid temperature raises error."""
    with pytest.raises(ValueError, match="temperature must be between"):
        MockProvider(model_name="test", temperature=3.0)


def test_cost_calculation():
    """Test cost calculation."""
    provider = MockProvider(model_name="test")
    cost = provider.calculate_cost(input_tokens=1000, output_tokens=1000)
    assert cost == 0.04  # (1 * 0.01) + (1 * 0.03)


def test_usage_tracking():
    """Test usage tracking accumulates."""
    provider = MockProvider(model_name="test")
    
    usage1 = provider.track_usage(100, 100)
    usage2 = provider.track_usage(200, 200)
    
    total = provider.get_total_usage()
    assert total.input_tokens == 300
    assert total.output_tokens == 300
    assert total.total_tokens == 600


def test_capability_detection():
    """Test capability detection."""
    provider = MockProvider(model_name="test")
    assert provider.supports_capability(ModelCapability.TEMPERATURE_CONTROL)
    assert provider.supports_capability(ModelCapability.STREAMING)