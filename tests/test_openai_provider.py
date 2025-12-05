"""
Tests for OpenAI provider implementation.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from src.core.llm.openai_provider import OpenAIProvider
from src.core.llm.base_provider import ModelCapability


class TestOpenAIProvider:
    """Test suite for OpenAIProvider."""
    
    def test_initialization_with_env_key(self):
        """Test provider initializes with API key from environment."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            provider = OpenAIProvider(model_name="gpt-4o")
            assert provider.model_name == "gpt-4o"
            assert provider.api_key == "sk-test123"
            assert provider.provider_name == "openai"
    
    def test_initialization_with_explicit_key(self):
        """Test provider initializes with explicit API key."""
        provider = OpenAIProvider(
            model_name="gpt-4o-mini",
            api_key="sk-explicit"
        )
        assert provider.api_key == "sk-explicit"
    
    def test_initialization_without_key_raises_error(self):
        """Test provider raises error when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key not found"):
                OpenAIProvider(model_name="gpt-4o")
    
    def test_gpt4o_model_info(self):
        """Test GPT-4o model capabilities and pricing."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            provider = OpenAIProvider(model_name="gpt-4o")
            info = provider.get_model_info()
            
            assert info.provider == "openai"
            assert info.context_window == 128000
            assert info.supports_streaming is True
            assert info.supports_function_calling is True
            assert info.supports_vision is True
            assert info.is_reasoning_model is False
            assert info.cost_per_1k_input_tokens == 0.0025  # $2.50 per 1M
            assert info.cost_per_1k_output_tokens == 0.0100  # $10.00 per 1M
    
    def test_o1_model_info(self):
        """Test o1-preview reasoning model capabilities."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            provider = OpenAIProvider(model_name="o1-preview")
            info = provider.get_model_info()
            
            assert info.is_reasoning_model is True
            assert info.supports_function_calling is False  # o1 doesn't support tools
            assert info.cost_per_1k_input_tokens == 0.0150  # $15.00 per 1M
            assert info.cost_per_1k_output_tokens == 0.0600  # $60.00 per 1M
    
    def test_reasoning_model_detection(self):
        """Test reasoning model detection."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            gpt4o = OpenAIProvider(model_name="gpt-4o")
            o1 = OpenAIProvider(model_name="o1-mini")
            
            assert gpt4o._is_reasoning_model() is False
            assert o1._is_reasoning_model() is True
    
    def test_capability_detection_gpt4o(self):
        """Test capability detection for GPT-4o."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            provider = OpenAIProvider(model_name="gpt-4o")
            
            assert provider.supports_capability(ModelCapability.TEMPERATURE_CONTROL) is True
            assert provider.supports_capability(ModelCapability.FUNCTION_CALLING) is True
            assert provider.supports_capability(ModelCapability.STREAMING) is True
            assert provider.supports_capability(ModelCapability.JSON_MODE) is True
    
    def test_capability_detection_o1(self):
        """Test capability detection for o1 reasoning models."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            provider = OpenAIProvider(model_name="o1-preview")
            
            assert provider.supports_capability(ModelCapability.TEMPERATURE_CONTROL) is False
            assert provider.supports_capability(ModelCapability.FUNCTION_CALLING) is False
            assert provider.supports_capability(ModelCapability.JSON_MODE) is False
    
    def test_cost_calculation(self):
        """Test cost calculation for different models."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            gpt4o = OpenAIProvider(model_name="gpt-4o")
            mini = OpenAIProvider(model_name="gpt-4o-mini")
            
            # GPT-4o: $2.50 input, $10.00 output per 1M tokens
            cost_gpt4o = gpt4o.calculate_cost(1000, 1000)  # 1k tokens each
            assert cost_gpt4o == pytest.approx(0.0125)  # (1 * 2.50 + 1 * 10.00) / 1000
            
            # GPT-4o-mini: $0.15 input, $0.60 output per 1M tokens
            cost_mini = mini.calculate_cost(1000, 1000)
            assert cost_mini == pytest.approx(0.00075)  # (1 * 0.15 + 1 * 0.60) / 1000
    
    def test_usage_tracking(self):
        """Test cumulative usage tracking."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            provider = OpenAIProvider(model_name="gpt-4o")
            
            # Track multiple usages
            usage1 = provider.track_usage(1000, 500)
            usage2 = provider.track_usage(2000, 1000)
            
            total = provider.get_total_usage()
            assert total.input_tokens == 3000
            assert total.output_tokens == 1500
            assert total.total_tokens == 4500
            assert total.estimated_cost > 0
    
    def test_base_model_name_extraction(self):
        """Test extracting base model name from versioned strings."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            provider = OpenAIProvider(model_name="gpt-4o-2024-11-20")
            
            base_name = provider._get_base_model_name()
            assert base_name == "gpt-4o"
            
            # Test with non-versioned model
            provider2 = OpenAIProvider(model_name="gpt-4o")
            assert provider2._get_base_model_name() == "gpt-4o"
    
    def test_client_creation_mocked(self):
        """Test client creation returns correct type (mocked)."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            provider = OpenAIProvider(model_name="gpt-4o")
            
            # Mock the client creation to avoid actual API calls
            with patch('src.core.llm.openai_provider.OpenAIChatCompletionClient') as mock_client:
                mock_instance = MagicMock()
                mock_client.return_value = mock_instance
                
                client = provider.create_model_client()
                
                # Verify client was created with correct params
                mock_client.assert_called_once()
                call_kwargs = mock_client.call_args.kwargs
                assert call_kwargs["model"] == "gpt-4o"
                assert call_kwargs["api_key"] == "sk-test"
                assert "temperature" in call_kwargs
    
    def test_repr(self):
        """Test string representation."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            provider = OpenAIProvider(model_name="gpt-4o")
            repr_str = repr(provider)
            
            assert "OpenAIProvider" in repr_str
            assert "gpt-4o" in repr_str
            assert "standard" in repr_str  # Model type


# Optional: Integration test (requires real API key, costs money)
@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or os.getenv("SKIP_API_TESTS") == "1",
    reason="Requires OPENAI_API_KEY and costs money"
)
class TestOpenAIProviderIntegration:
    """Integration tests with real OpenAI API (optional)."""
    
    @pytest.mark.asyncio
    async def test_real_api_call(self):
        """Test real API call with GPT-4o-mini (cheapest model)."""
        from autogen_core.models import UserMessage
        
        provider = OpenAIProvider(model_name="gpt-4o-mini", max_tokens=50)
        client = provider.create_model_client()
        
        try:
            # Make a cheap test call
            response = await client.create([
                UserMessage(content="Say 'Hello' and nothing else.", source="user")
            ])
            
            assert response.content is not None
            assert len(response.content) > 0
            
            print(f"\n✅ API call successful!")
            print(f"   Response: {response.content[:100]}")
            
            # Note: AutoGen 0.4 OpenAIChatCompletionClient doesn't track usage automatically
            # You'd need to extract from response.usage if needed
            if hasattr(response, 'usage') and response.usage:
                print(f"   Tokens: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
        
        finally:
            # AutoGen 0.4 clients don't have close() method
            # Cleanup happens automatically
            pass