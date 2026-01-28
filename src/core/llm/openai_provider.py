"""
OpenAI LLM Provider Implementation.

This module provides concrete implementation of BaseLLMProvider for OpenAI models,
including GPT-5, GPT-4.1 (1M Context), and o3 reasoning models.
"""

import os
from typing import Optional, Dict, Any

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily

from .base_provider import BaseLLMProvider, ModelInfo, ModelCapability


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider supporting GPT-5, GPT-4.1, and o3 model families.
    
    Supports:
    - gpt-5.2: 2026 Flagship (400k context, advanced reasoning)
    - gpt-5-mini: 2026 High-efficiency model
    - gpt-4.1: 1-Million token context specialist
    - o3-preview/mini: Next-gen reasoning models (supports tools)
    """
    
    # Model pricing (Jan 2026, per 1M tokens)
    MODEL_PRICING = {
        "gpt-5.2": {"input": 1.75, "output": 14.00},
        "gpt-5-mini": {"input": 0.125, "output": 1.00},
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "o3-preview": {"input": 8.00, "output": 32.00},
        "o3-mini": {"input": 1.50, "output": 6.00},
        "gpt-4o": {"input": 1.00, "output": 4.00},
        "gpt-4o-mini": {"input": 0.10, "output": 0.40},
    }
    
    # Reasoning models (Support max_completion_tokens)
    REASONING_MODELS = {
        "o3-preview", "o3-mini", "o3-full", 
        "o1-preview", "o1-mini"
    }
    
    # Context windows
    CONTEXT_WINDOWS = {
        "gpt-5.2": 400000,
        "gpt-5-mini": 400000,
        "gpt-4.1": 1048576,
        "o3-preview": 256000,
        "o3-mini": 256000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 150000,
    }

    # Metadata for AutoGen's internal registry (Prevents KeyError/ValueError)
    MODEL_METADATA = {
        "gpt-5.2": {"vision": True, "function_calling": True, "json_output": True, "family": ModelFamily.GPT_4},
        "gpt-5.2-2025-12-11": {"vision": True, "function_calling": True, "json_output": True, "family": ModelFamily.GPT_4},
        "gpt-4.1": {"vision": True, "function_calling": True, "json_output": True, "family": ModelFamily.GPT_4},
        "o3-preview": {"vision": True, "function_calling": True, "json_output": True, "family": ModelFamily.GPT_4},
        "o3-mini": {"vision": True, "function_calling": True, "json_output": True, "family": ModelFamily.GPT_4},
    }
    
    def __init__(
        self,
        model_name: str = "gpt-5.2",
        temperature: Optional[float] = 0.3,
        max_tokens: Optional[int] = 4096,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.organization = organization
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found.")
        
        if not self._is_known_model(model_name):
            print(f"Warning: Unknown model '{model_name}'. Pricing may be inaccurate.")
        
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    def create_model_client(self) -> OpenAIChatCompletionClient:
        client_kwargs = {
            "model": self.model_name,
            "api_key": self.api_key,
        }

        # Access metadata to handle parameter naming
        base_model = self._get_base_model_name()
        if base_model in self.MODEL_METADATA:
            client_kwargs["model_info"] = self.MODEL_METADATA[base_model]

        # --- 2026 PARAMETER LOGIC ---
        # Newer models (GPT-5 series, GPT-4.1, and o-series) require max_completion_tokens
        uses_new_params = any(m in base_model for m in ["gpt-5", "gpt-4.1", "o3", "o4", "o1"])
        is_reasoning = self._is_reasoning_model()

        if self.max_tokens:
            if uses_new_params:
                client_kwargs["max_completion_tokens"] = self.max_tokens
            else:
                client_kwargs["max_tokens"] = self.max_tokens

        # Standard temperature logic
        # Note: GPT-5 supports temperature, but reasoning models (o-series) do not.
        if not is_reasoning and self.temperature is not None:
            client_kwargs["temperature"] = self.temperature

        # Custom settings (proxies, orgs, etc.)
        if self.base_url: client_kwargs["base_url"] = self.base_url
        if self.organization: client_kwargs["organization"] = self.organization
        client_kwargs.update(self.extra_params)

        return OpenAIChatCompletionClient(**client_kwargs)


    def get_model_info(self) -> ModelInfo:
        is_reasoning = self._is_reasoning_model()
        base_model = self._get_base_model_name()
        
        pricing = self.MODEL_PRICING.get(base_model, {"input": 0.0, "output": 0.0})
        context_window = self.CONTEXT_WINDOWS.get(base_model, 128000)
        
        # 2026: o3 models support function calling; older o1 models usually do not.
        supports_functions = ("o3" in base_model) or (not is_reasoning)
        supports_vision = any(m in base_model for m in ["gpt-5", "gpt-4.1", "gpt-4o", "o3"])

        return ModelInfo(
            name=self.model_name,
            provider=self.provider_name,
            context_window=context_window,
            supports_streaming=True,
            supports_function_calling=supports_functions,
            supports_vision=supports_vision,
            is_reasoning_model=is_reasoning,
            cost_per_1k_input_tokens=pricing["input"] / 1000,
            cost_per_1k_output_tokens=pricing["output"] / 1000,
        )

    def _is_reasoning_model(self) -> bool:
        return any(rm in self.model_name for rm in self.REASONING_MODELS)
    
    def _is_known_model(self, model_name: str) -> bool:
        base_name = self._get_base_model_name(model_name)
        return base_name in self.MODEL_PRICING
    
    def _get_base_model_name(self, model_name: Optional[str] = None) -> str:
        name = model_name or self.model_name
        valid_years = ["-2023-", "-2024-", "-2025-", "-2026-"]
        
        if any(year in name for year in valid_years):
            parts = name.split("-")
            for i, part in enumerate(parts):
                if part.isdigit() and len(part) == 4:
                    return "-".join(parts[:i])
        return name
    
    def __repr__(self) -> str:
        model_type = "reasoning" if self._is_reasoning_model() else "standard"
        return f"OpenAIProvider(model={self.model_name}, type={model_type})"
