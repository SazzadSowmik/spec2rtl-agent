# Spec2RTL-Agent

**Automated Hardware Code Generation from Complex Specifications Using LLM Agent Systems**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![AutoGen 0.4](https://img.shields.io/badge/AutoGen-0.4-green.svg)](https://microsoft.github.io/autogen/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of the [Spec2RTL-Agent paper](https://arxiv.org/abs/2506.13905v2) - an LLM-based multi-agent system for end-to-end RTL generation from specification documents using AutoGen 0.4 framework.

---

## 🎯 Project Overview
```
Specification PDF → Understanding Module → Coding Module → Reflection Module → RTL Code
                    (Summarize, Decompose)  (Progressive)   (Adaptive Debug)
```

### Key Features

- 🤖 **Multi-Agent System**: AutoGen 0.4 async actor-based architecture
- 🔄 **Multi-LLM Support**: OpenAI (GPT-4o, o1), Anthropic (Claude), Local models
- 📄 **Document Processing**: Multi-modal PDF extraction (text, tables, figures)
- 💰 **Cost Tracking**: Built-in token usage and cost monitoring
- 🐳 **Dockerized**: Consistent development environment
- ✅ **Type-Safe**: Pydantic models throughout

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenAI API key (for GPT-4o/o1)
- Optional: Anthropic API key (for Claude)

### Setup
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/spec2rtl-agent.git
cd spec2rtl-agent

# Copy environment template
cp .env.example .env

# Add your API keys to .env
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Build and start container
docker-compose build
docker-compose up -d

# Run hello world
docker-compose exec spec2rtl python main.py

# Run tests
docker-compose exec spec2rtl pytest tests/ -v
```

### Expected Output
```
╭─────────────────── Spec2RTL-Agent ───────────────────╮
│ Hello World from Spec2RTL-Agent! 🚀                  │
│                                                      │
│ AutoGen 0.4 Multi-Agent System for RTL Generation    │
│ Ready to transform specifications into hardware code │
╰──────────────────────────────────────────────────────╯
✅ System initialized successfully!
📝 Next: Implement Understanding Module
```

---

## 📁 Project Structure
```
spec2rtl-agent/
├── config/                     # Configuration files
│   ├── llm_config.py          # LLM provider settings
│   └── agent_prompts.py       # Agent system prompts
├── src/
│   ├── core/                  # Core utilities
│   │   ├── llm/               # ✅ LLM provider abstraction
│   │   │   ├── base_provider.py      # Abstract interface
│   │   │   ├── openai_provider.py    # 🚧 OpenAI implementation
│   │   │   ├── anthropic_provider.py # ⏳ Anthropic (planned)
│   │   │   ├── local_provider.py     # ⏳ Local models (planned)
│   │   │   └── provider_factory.py   # ⏳ Factory pattern
│   │   ├── document_loader.py # ⏳ PDF extraction
│   │   └── data_models.py     # ⏳ Pydantic schemas
│   ├── agents/                # Agent implementations
│   │   ├── understanding/     # 🚧 Phase 1 (Current)
│   │   │   ├── summarization_agent.py
│   │   │   ├── decomposer_agent.py
│   │   │   ├── description_agent.py
│   │   │   └── verifier_agent.py
│   │   ├── coding/            # ⏳ Phase 2 (Planned)
│   │   └── reflection/        # ⏳ Phase 3 (Planned)
│   └── orchestration/         # Agent coordination
│       └── understanding_pipeline.py
├── data/                      # Input/Output data
│   ├── input/specs/          # Specification PDFs
│   ├── processed/sections/   # Extracted sections
│   └── output/summaries/     # Generated outputs
├── tests/                    # ✅ Unit tests (5/5 passing)
├── notebooks/                # Jupyter notebooks
├── Dockerfile                # ✅ Docker setup
├── docker-compose.yml        # ✅ Development environment
└── main.py                   # ✅ Entry point
```

**Legend:** ✅ Complete | 🚧 In Progress | ⏳ Planned

---

## 📊 Implementation Progress

### Phase 1: Understanding & Reasoning Module (IN PROGRESS)

**Goal:** Transform unstructured spec PDFs into structured implementation plans

#### 1.1 Core Infrastructure ✅

- [x] **LLM Provider Abstraction**
  - [x] Base provider interface with cost tracking
  - [x] Model capability detection (GPT-4 vs o1)
  - [x] Usage metrics and token counting
  - [x] Unit tests (5/5 passing)
- [ ] **OpenAI Provider** 🚧 (Next)
  - [ ] GPT-4o implementation
  - [ ] o1 reasoning model support
  - [ ] API key validation
  - [ ] Real API call testing
- [ ] **Data Models** ⏳
  - [ ] SpecSection schema
  - [ ] SectionSummary schema
  - [ ] ImplementationPlan schema
- [ ] **Document Loader** ⏳
  - [ ] PDF text extraction
  - [ ] Section boundary detection
  - [ ] Multi-modal content handling

#### 1.2 Agent Implementation ⏳

- [ ] Summarization Agent (First agent)
- [ ] Decomposer Agent
- [ ] Description Agent
- [ ] Verifier Agent
- [ ] Orchestration Pipeline

### Phase 2: Coding Module ⏳

- Progressive coding (Pseudocode → Python → C++)
- Prompt optimization
- Code verification

### Phase 3: Reflection Module ⏳

- Error analysis
- Adaptive debugging
- HLS integration

---

## 🛠️ Development

### Running Tests
```bash
# Run all tests
docker-compose exec spec2rtl pytest tests/ -v

# Run specific test file
docker-compose exec spec2rtl pytest tests/test_base_provider.py -v

# Run with coverage
docker-compose exec spec2rtl pytest --cov=src tests/
```

### Code Quality
```bash
# Format code
docker-compose exec spec2rtl black src/ tests/

# Sort imports
docker-compose exec spec2rtl isort src/ tests/

# Lint
docker-compose exec spec2rtl flake8 src/ tests/

# Type check
docker-compose exec spec2rtl mypy src/
```

### Interactive Development
```bash
# Python REPL
docker-compose exec spec2rtl python

# IPython shell
docker-compose exec spec2rtl ipython

# Jupyter Lab
docker-compose exec spec2rtl jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
# Then visit: http://localhost:8888
```

---

## 🏗️ Architecture Highlights

### Multi-LLM Provider System
```python
from src.core.llm.base_provider import BaseLLMProvider
from src.core.llm.openai_provider import OpenAIProvider

# Create provider
provider = OpenAIProvider(
    model_name="gpt-4o",
    temperature=0.3,
    max_tokens=4096
)

# Get model client for AutoGen
model_client = provider.create_model_client()

# Track costs automatically
usage = provider.get_total_usage()
print(f"Total cost: ${usage.estimated_cost:.4f}")
```

### Cost Tracking

Every LLM call is automatically tracked:
- Input/output token counts
- Estimated costs (model-specific pricing)
- Cumulative usage per provider
- Per-request metrics

---

## 📚 Documentation

- [AutoGen 0.4 Docs](https://microsoft.github.io/autogen/stable/)
- [Spec2RTL Paper](https://arxiv.org/abs/2506.13905v2)
- [AES Specification](./data/input/specs/AES_Spec.pdf)

---

## 🤝 Contributing

1. Follow the existing code structure
2. Write tests for new features
3. Use type hints throughout
4. Run code quality tools before committing
5. Make atomic commits with clear messages

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

Based on "Spec2RTL-Agent: Automated Hardware Code Generation from Complex Specifications Using LLM Agent Systems" by Yu et al. (Nvidia Research, Georgia Tech, Cadence)

---

## 📞 Contact

For questions or issues, please open a GitHub issue.

---

**Status:** 🚧 Active Development | **Last Updated:** December 2024