import asyncio
from rich.console import Console
from rich.panel import Panel

console = Console()


async def main():
    """Main entry point"""
    
    console.print(Panel.fit(
        "[bold green]Hello World from Spec2RTL-Agent! 🚀[/bold green]\n\n"
        "[cyan]AutoGen 0.4 Multi-Agent System for RTL Generation[/cyan]\n"
        "[dim]Ready to transform specifications into hardware code[/dim]",
        title="Spec2RTL-Agent",
        border_style="green"
    ))
    
    # Test async functionality
    await asyncio.sleep(0.1)
    
    console.print("\n✅ [green]System initialized successfully![/green]")
    console.print("📝 [yellow]Next: Implement Understanding Module[/yellow]\n")


if __name__ == "__main__":
    asyncio.run(main())