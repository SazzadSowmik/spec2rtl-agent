#!/bin/bash
set -e

echo "🚀 Starting Spec2RTL-Agent Container..."

# Validate environment variables
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  Warning: No API keys found in environment"
    echo "   Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file"
fi

# Check Python version
python --version

# Check installed packages
echo "📦 Installed packages:"
pip list | grep -E "autogen|openai|anthropic|pydantic"

echo "✅ Container ready!"
echo "   Run commands with: docker-compose exec spec2rtl <command>"

# Execute the main command
exec "$@"