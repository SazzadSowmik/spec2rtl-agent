"""
Spec2RTL-Agent - Main Entry Point
"""

import asyncio
from rich.console import Console
from src.core.logging_config import setup_logging

console = Console()


async def main():
    """Run Understanding Agent test."""
    # Setup logging
    logger = setup_logging(log_dir="logs")
    
    console.print("\n[bold blue]🚀 Spec2RTL-Agent - Understanding Module Test[/bold blue]\n")
    
    # Import and run test
    from src.orchestration.understanding_pipeline import test_understanding_pipeline
    from src.agents.understanding.decomposer_agent import test_decomposer_agent
    
    await test_understanding_pipeline()
    # await test_decomposer_agent()
    
    console.print("\n[bold green]✅ Test complete![/bold green]")
    console.print("\n[yellow]📁 Check logs/ directory for detailed logs[/yellow]")
    console.print("[yellow]📁 Check data/output/summaries/ for saved summaries[/yellow]\n")


if __name__ == "__main__":
    asyncio.run(main())