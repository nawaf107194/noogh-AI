#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cabinet Test Script - Test AI-Powered Ministers
================================================

Tests each minister with domain-specific queries to verify they're
using Meta-Llama-3-8B intelligence.
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


async def test_cabinet():
    """Test all ministers with domain-specific queries."""
    
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]🏛️ Testing AI-Powered Cabinet Ministers[/bold cyan]")
    console.print("=" * 70 + "\n")
    
    # Initialize President
    console.print("[yellow]Initializing President with Smart Cabinet...[/yellow]\n")
    
    try:
        from src.government.president import President
        
        president = President(verbose=True)
        
        console.print("[green]✅ President and Cabinet ready![/green]\n")
        
        # Check if brain is available
        if not hasattr(president, 'brain') or president.brain is None:
            console.print("[red]❌ Neural Core not available![/red]")
            console.print("Ministers will have limited capability.")
            console.print("Run: python scripts/setup_local_model.py\n")
        
        # Display cabinet
        table = Table(title="Smart Cabinet")
        table.add_column("Minister", style="cyan")
        table.add_column("Description", style="yellow")
        table.add_column("Brain Status", style="green")
        
        for name, minister in president.cabinet.items():
            has_brain = "✅ AI-Powered" if minister.brain else "❌ No Brain"
            table.add_row(
                minister.name,
                minister.description,
                has_brain
            )
        
        console.print(table)
        console.print()
        
    except Exception as e:
        console.print(f"[red]❌ Error initializing: {e}[/red]")
        import traceback
        traceback.print_exc()
        return
    
    # Test queries for each minister
    tests = [
        {
            "minister": "education",
            "query": "Explain what is photosynthesis in simple terms",
            "expected": "Educational explanation with examples"
        },
        {
            "minister": "security",
            "query": "SELECT * FROM users WHERE 1=1; DROP TABLE users;",
            "expected": "Security threat detection"
        },
        {
            "minister": "development",
            "query": "A function that calculates fibonacci numbers",
            "expected": "Python code with documentation"
        },
    ]
    
    for i, test in enumerate(tests, 1):
        minister_name = test["minister"]
        query = test["query"]
        
        console.print(f"[bold cyan]Test {i}: {minister_name.upper()} Minister[/bold cyan]")
        console.print(f"[yellow]Query:[/yellow] {query}")
        console.print(f"[dim]Expected:[/dim] {test['expected']}\n")
        
        try:
            # Get minister
            minister = president.cabinet.get(minister_name)
            
            if not minister:
                console.print(f"[red]❌ Minister '{minister_name}' not found![/red]\n")
                continue
            
            # Execute task
            result = await minister.execute_task(query)
            
            # Display result
            if result.get("success"):
                response = result.get("response", "No response")
                domain = result.get("domain", "unknown")
                metadata = result.get("metadata", {})
                
                console.print(Panel(
                    f"[bold]Response:[/bold]\n{response}\n\n"
                    f"[dim]Domain:[/dim] {domain}\n"
                    f"[dim]Metadata:[/dim] {metadata}",
                    title=f"[green]{minister.name} Response[/green]",
                    border_style="green"
                ))
            else:
                error = result.get("error", "Unknown error")
                console.print(f"[red]❌ Failed: {error}[/red]")
            
            # Show stats
            stats = minister.get_stats()
            console.print(f"[dim]   Processed: {stats['tasks_processed']} | "
                         f"Success Rate: {stats['success_rate']:.1%}[/dim]\n")
        
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]\n")
            import traceback
            traceback.print_exc()
    
    console.print("=" * 70)
    console.print("[bold green]✨ Cabinet test complete![/bold green]")
    console.print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_cabinet())
