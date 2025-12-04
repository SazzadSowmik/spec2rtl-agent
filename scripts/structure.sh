#!/bin/bash
# Create Spec2RTL-Agent Project Structure

# Root directory
# mkdir -p spec2rtl-agent
# cd spec2rtl-agent

# Config directory
mkdir -p config
touch config/__init__.py
touch config/llm_config.py
touch config/agent_prompts.py

# Source directories
mkdir -p src/core/llm
mkdir -p src/agents/understanding
mkdir -p src/agents/coding
mkdir -p src/agents/reflection
mkdir -p src/orchestration

# Core files
touch src/__init__.py
touch src/core/__init__.py
touch src/core/document_loader.py
touch src/core/data_models.py

# LLM Provider files
touch src/core/llm/__init__.py
touch src/core/llm/base_provider.py
touch src/core/llm/openai_provider.py
touch src/core/llm/anthropic_provider.py
touch src/core/llm/local_provider.py
touch src/core/llm/provider_factory.py

# Agent files
touch src/agents/__init__.py
touch src/agents/base_agent.py
touch src/agents/understanding/__init__.py
touch src/agents/understanding/summarization_agent.py
touch src/agents/understanding/decomposer_agent.py
touch src/agents/understanding/description_agent.py
touch src/agents/understanding/verifier_agent.py
touch src/agents/coding/__init__.py
touch src/agents/reflection/__init__.py

# Orchestration
touch src/orchestration/__init__.py
touch src/orchestration/understanding_pipeline.py

# Data directories
mkdir -p data/input/specs
mkdir -p data/processed/sections
mkdir -p data/output/summaries

# Create .gitkeep for empty directories
touch data/input/specs/.gitkeep
touch data/processed/sections/.gitkeep
touch data/output/summaries/.gitkeep

# Tests
mkdir -p tests
touch tests/__init__.py
touch tests/test_providers.py
touch tests/test_understanding_agent.py

# Notebooks (optional)
mkdir -p notebooks
touch notebooks/demo_understanding.ipynb

# Root files
touch README.md
touch requirements.txt
touch .env.example
touch .gitignore
touch setup.py

echo "✅ Project structure created successfully!"