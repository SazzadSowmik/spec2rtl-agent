# Spec2RTL-Agent

Automated Hardware Code Generation from Complex Specifications Using LLM Agent Systems

## Overview

Implementation of the [Spec2RTL-Agent paper](https://arxiv.org/abs/2506.13905v2) - an LLM-based multi-agent system for end-to-end RTL generation from specification documents.

## Architecture
```
Specification PDF → Understanding Module → Coding Module → Reflection Module → RTL Code
```

### Current Implementation Status

- [x] Project Structure
- [ ] **Phase 1: Understanding Module** (IN PROGRESS)
  - [ ] Summarization Agent
  - [ ] Decomposer Agent
  - [ ] Description Agent
  - [ ] Verifier Agent
- [ ] Phase 2: Coding Module
- [ ] Phase 3: Reflection Module

## Features

- 🔄 Multi-LLM Support (OpenAI, Anthropic, Local Models)
- 🏗️ Modular Architecture (Easy to extend)
- 🤖 AutoGen Framework Integration
- 📄 Multi-modal Document Processing
- ✅ Type-safe with Pydantic

## Installation
```bash
# Clone repository
git clone https://github.com/yourusername/spec2rtl-agent.git
cd spec2rtl-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start
```python
# Coming soon - Summarization Agent demo
```

## Project Structure
```
spec2rtl-agent/
├── config/              # Configuration files
├── src/
│   ├── core/           # Core utilities
│   │   └── llm/        # LLM provider abstraction
│   ├── agents/         # Agent implementations
│   │   ├── understanding/
│   │   ├── coding/
│   │   └── reflection/
│   └── orchestration/  # Agent coordination
├── data/               # Input/Output data
├── tests/              # Unit tests
└── notebooks/          # Jupyter notebooks
```

## Development Roadmap

### Phase 1: Understanding & Reasoning Module
- [ ] Document Loader
- [ ] Summarization Agent
- [ ] Decomposer Agent
- [ ] Description Agent
- [ ] Verifier Agent

### Phase 2: Coding Module
- [ ] Progressive Coding (Pseudocode → Python → C++)
- [ ] Prompt Optimization
- [ ] Code Verification

### Phase 3: Reflection Module
- [ ] Error Analysis
- [ ] Adaptive Debugging
- [ ] HLS Integration

## Contributing

Contributions welcome! Please check issues or create new ones.

## License

MIT License

## Citation
```bibtex
@article{yu2025spec2rtl,
  title={Spec2RTL-Agent: Automated Hardware Code Generation from Complex Specifications Using LLM Agent Systems},
  author={Yu, Zhongzhi and Liu, Mingjie and others},
  journal={arXiv preprint arXiv:2506.13905},
  year={2025}
}
```

## Acknowledgments

Based on the paper "Spec2RTL-Agent" by Yu et al. (Nvidia Research, Georgia Tech, Cadence)