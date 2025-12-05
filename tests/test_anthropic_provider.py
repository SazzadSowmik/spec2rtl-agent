"""
Tests for Anthropic provider implementation.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from src.core.llm.anthropic_provider import AnthropicProvider
from src.core.llm.base_provider import ModelCapability


class TestAnthropicProvider:
    """Test suite for AnthropicProvider."""
    
    def test_initialization_with_env_key(self):
        """Test provider initializes with API key from environment."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test123"}):
            provider = AnthropicProvider(model_name="claude-3-7-sonnet-20250219")
            assert provider.model_name == "claude-3-7-sonnet-20250219"
            assert provider.api_key == "sk-ant-test123"
            assert provider.provider_name == "anthropic"
    
    def test_initialization_with_explicit_key(self):
        """Test provider initializes with explicit API key."""
        provider = AnthropicProvider(
            model_name="claude-3-5-haiku-20241022",
            api_key="sk-ant-explicit"
        )
        assert provider.api_key == "sk-ant-explicit"
    
    def test_initialization_without_key_raises_error(self):
        """Test provider raises error when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key not found"):
                AnthropicProvider(model_name="claude-3-7-sonnet-20250219")
    
    def test_max_tokens_default(self):
        """Test max_tokens defaults to 4096 if not provided."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            provider = AnthropicProvider(
                model_name="claude-3-7-sonnet-20250219",
                max_tokens=None  # Explicitly pass None
            )
            assert provider.max_tokens == 4096
    
    def test_claude_37_model_info(self):
        """Test Claude 3.7 Sonnet model capabilities and pricing."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            provider = AnthropicProvider(model_name="claude-3-7-sonnet-20250219")
            info = provider.get_model_info()
            
            assert info.provider == "anthropic"
            assert info.context_window == 200000
            assert info.supports_streaming is True
            assert info.supports_function_calling is True
            assert info.supports_vision is True
            assert info.is_reasoning_model is False
            assert info.cost_per_1k_input_tokens == 0.003  # $3.00 per 1M
            assert info.cost_per_1k_output_tokens == 0.015  # $15.00 per 1M
    
    def test_claude_haiku_model_info(self):
        """Test Claude Haiku (cheapest) model pricing."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            provider = AnthropicProvider(model_name="claude-3-5-haiku-20241022")
            info = provider.get_model_info()
            
            assert info.cost_per_1k_input_tokens == 0.0008  # $0.80 per 1M
            assert info.cost_per_1k_output_tokens == 0.004  # $4.00 per 1M
    
    def test_claude_opus_model_info(self):
        """Test Claude Opus (most expensive) model pricing."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            provider = AnthropicProvider(model_name="claude-3-opus-20240229")
            info = provider.get_model_info()
            
            assert info.cost_per_1k_input_tokens == 0.015  # $15.00 per 1M
            assert info.cost_per_1k_output_tokens == 0.075  # $75.00 per 1M
    
    def test_capability_detection(self):
        """Test capability detection for Claude models."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            provider = AnthropicProvider(model_name="claude-3-7-sonnet-20250219")
            
            assert provider.supports_capability(ModelCapability.TEMPERATURE_CONTROL) is True
            assert provider.supports_capability(ModelCapability.FUNCTION_CALLING) is True
            assert provider.supports_capability(ModelCapability.STREAMING) is True
            assert provider.supports_capability(ModelCapability.VISION) is True
            assert provider.supports_capability(ModelCapability.JSON_MODE) is True
    
    def test_cost_calculation(self):
        """Test cost calculation for Claude models."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            sonnet = AnthropicProvider(model_name="claude-3-7-sonnet-20250219")
            haiku = AnthropicProvider(model_name="claude-3-5-haiku-20241022")
            
            # Sonnet: $3.00 input, $15.00 output per 1M tokens
            cost_sonnet = sonnet.calculate_cost(1000, 1000)
            assert cost_sonnet == pytest.approx(0.018)  # (1 * 3.00 + 1 * 15.00) / 1000
            
            # Haiku: $0.80 input, $4.00 output per 1M tokens
            cost_haiku = haiku.calculate_cost(1000, 1000)
            assert cost_haiku == pytest.approx(0.0048)  # (1 * 0.80 + 1 * 4.00) / 1000
    
    def test_usage_tracking(self):
        """Test cumulative usage tracking."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            provider = AnthropicProvider(model_name="claude-3-7-sonnet-20250219")
            
            # Track multiple usages
            provider.track_usage(1000, 500)
            provider.track_usage(2000, 1000)
            
            total = provider.get_total_usage()
            assert total.input_tokens == 3000
            assert total.output_tokens == 1500
            assert total.total_tokens == 4500
            assert total.estimated_cost > 0
    
    def test_client_creation_mocked(self):
        """Test client creation returns correct type (mocked)."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            provider = AnthropicProvider(model_name="claude-3-7-sonnet-20250219")
            
            # Mock the client creation
            with patch('src.core.llm.anthropic_provider.AnthropicChatCompletionClient') as mock_client:
                mock_instance = MagicMock()
                mock_client.return_value = mock_instance
                
                client = provider.create_model_client()
                
                # Verify client was created with correct params
                mock_client.assert_called_once()
                call_kwargs = mock_client.call_args.kwargs
                assert call_kwargs["model"] == "claude-3-7-sonnet-20250219"
                assert call_kwargs["api_key"] == "sk-ant-test"
                assert call_kwargs["max_tokens"] == 4096
                assert "temperature" in call_kwargs
    
    def test_repr(self):
        """Test string representation."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            provider = AnthropicProvider(model_name="claude-3-7-sonnet-20250219")
            repr_str = repr(provider)
            
            assert "AnthropicProvider" in repr_str
            assert "claude-3-7-sonnet-20250219" in repr_str


# Optional: Integration test (skipped by default)
@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY") or os.getenv("SKIP_API_TESTS") == "1",
    reason="Requires ANTHROPIC_API_KEY and costs money"
)
class TestAnthropicProviderIntegration:
    """Integration tests with real Anthropic API (optional)."""
    
    @pytest.mark.asyncio
    async def test_real_api_call(self):
        """Test real API call with Claude Haiku (cheapest model)."""
        from autogen_core.models import UserMessage
        
        provider = AnthropicProvider(
            model_name="claude-3-5-haiku-20241022",
            max_tokens=50
        )
        client = provider.create_model_client()
        
        try:
            # Make a cheap test call
            response = await client.create([
                UserMessage(content="Say 'Hello' and nothing else.", source="user")
            ])
            
            assert response.content is not None
            assert len(response.content) > 0
            
            print(f"\n✅ Anthropic API call successful!")
            print(f"   Response: {response.content[:100]}")
            
            if hasattr(response, 'usage') and response.usage:
                print(f"   Tokens: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
        
        finally:
            # Cleanup happens automatically
            pass