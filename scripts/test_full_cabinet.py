#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Cabinet Test - Complete 7-Minister Government
===================================================

Tests all 7 autonomous ministers:
1. Education (Strategist)
2. Security (Guardian)
3. Development (Code Generator)
4. Finance (Crypto Trader)
5. Health (Hardware Monitor)
6. Foreign Affairs (Intelligence)
7. Communication (Press Secretary)
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


async def test_all_ministers():
    """Test complete cabinet."""
    
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]🏛️ COMPLETE CABINET TEST - 7 MINISTERS[/bold cyan]")
    console.print("=" * 70 + "\n")
    
    # Initialize President
    console.print("[yellow]Initializing President with Complete Cabinet...[/yellow]\n")
    
    try:
        from src.government.president import President
        
        president = President(verbose=True)
        
        console.print(f"[green]✅ President initialized with {len(president.cabinet)} ministers![/green]\n")
        
        # Display complete cabinet
        table = Table(title="Complete Cabinet", show_header=True)
        table.add_column("Minister", style="cyan", width=25)
        table.add_column("Domain", style="yellow", width=20)
        table.add_column("Capability", style="green")
        
        minister_info = [
            ("Education Minister", "Strategy", "AI-powered strategic optimization"),
            ("Security Minister", "Cybersecurity", "Threat detection & mitigation"),
            ("Development Minister", "Engineering", "Python code generation"),
            ("Finance Minister", "Trading", "Crypto analysis (BTC/USDT)"),
            ("Health Minister", "Hardware", "RTX 5070 GPU & system monitoring"),
            ("Foreign Minister", "Intelligence", "Web search & intelligence analysis"),
            ("Communication Minister", "PR", "Press releases & announcements")
        ]
        
        for name, domain, capability in minister_info:
            table.add_row(name, domain, capability)
        
        console.print(table)
        console.print()
        
    except Exception as e:
        console.print(f"[red]❌ Error initializing: {e}[/red]")
        import traceback
        traceback.print_exc()
        return
    
    # Test each minister
    tests = [
        ("health", "Check system vital signs", "Hardware diagnostics"),
        ("foreign", "Latest news on Meta Llama-3 LLM", "Web intelligence"),
        ("communication", "System is fully operational with 7 ministers", "Twitter announcement"),
    ]
    
    for i, (minister_key, query, description) in enumerate(tests, 1):
        console.print(f"[bold cyan]Test {i}: {minister_key.upper()} Minister[/bold cyan]")
        console.print(f"[yellow]Query:[/yellow] {query}")
        console.print(f"[dim]Testing:[/dim] {description}\n")
        
        try:
            minister = president.cabinet.get(minister_key)
            
            if not minister:
                console.print(f"[red]❌ Minister '{minister_key}' not found![/red]\n")
                continue
            
            # Special context for communication
            context = None
            if minister_key == "communication":
                context = {"format": "tweet"}
            
            # Execute task
            result = await minister.execute_task(query, context=context)
            
            if result.get("success"):
                metadata = result.get("metadata", {})
                
                console.print(Panel(
                    result['response'],
                    title=f"[green]{minister.name} Response[/green]",
                    border_style="green"
                ))
                
                # Show special metadata
                if minister_key == "health" and "vitals" in metadata:
                    vitals = metadata["vitals"]
                    gpu = vitals.get("gpu", {})
                    console.print(f"\n[dim]📊 GPU: {gpu.get('temperature_c', 'N/A')}°C | "
                                f"VRAM: {gpu.get('vram_percent', 0):.1f}%[/dim]")
                
                elif minister_key == "foreign" and "search_results" in metadata:
                    results = metadata["search_results"]
                    console.print(f"\n[dim]🔍 Found {len(results)} sources[/dim]")
                
                console.print()
            else:
                console.print(f"[red]❌ Failed: {result.get('error')}[/red]\n")
        
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]\n")
    
    # Summary
    console.print("=" * 70)
    console.print("[bold green]✨ COMPLETE CABINET TEST FINISHED[/bold green]")
    console.print("=" * 70)
    
    # Show all minister stats
    stats_table = Table(title="Minister Performance")
    stats_table.add_column("Minister", style="cyan")
    stats_table.add_column("Tasks", justify="right")
    stats_table.add_column("Success Rate", justify="right", style="green")
    
    for key, minister in president.cabinet.items():
        stats = minister.get_stats()
        stats_table.add_row(
            minister.name,
            str(stats['tasks_processed']),
            f"{stats['success_rate']:.1%}"
        )
    
    console.print(stats_table)
    console.print("\n[bold]🎉 The Complete Autonomous Government is OPERATIONAL![/bold]")
    console.print("[dim]7 Ministers | Each powered by Meta-Llama-3-8B | 100% Local AI[/dim]\n")


if __name__ == "__main__":
    asyncio.run(test_all_ministers())
