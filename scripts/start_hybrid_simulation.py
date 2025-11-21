#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Simulation - News + Volume Driven Trading
=================================================

PAPER TRADING ONLY: Simulates spot + futures trades.
News monitoring + volume scanning + multi-minister coordination.
"""

import asyncio
import json
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live

console = Console()


async def run_news_loop(finance_minister, foreign_minister):
    """News monitoring loop - triggers finance analysis."""
    console.print("[yellow]📰 News loop starting...[/yellow]")
    
    while True:
        try:
            # Monitor news
            result = await foreign_minister.execute_task("Monitor crypto news")
            
            if result.get("success"):
                metadata = result.get("metadata", {})
                triggers = metadata.get("triggers", [])
                
                # If high-impact news found, trigger finance analysis
                for trigger in triggers:
                    symbol = trigger.get("symbol")
                    reason = trigger.get("reason")
                    
                    console.print(f"\n[bold red]🚨 NEWS TRIGGER: {symbol}[/bold red]")
                    console.print(f"[yellow]Reason: {reason}[/yellow]\n")
                    
                    # Trigger finance minister
                    await finance_minister.execute_task(
                        f"Analyze {symbol}",
                        context={"symbol": symbol, "trigger": f"News: {reason}"}
                    )
            
            # Check every 5 minutes
            await asyncio.sleep(300)
        
        except Exception as e:
            console.print(f"[red]News loop error: {e}[/red]")
            await asyncio.sleep(60)


async def run_volume_loop(finance_minister):
    """Volume scanning loop - hunts for spikes."""
    console.print("[yellow]📊 Volume loop starting...[/yellow]")
    
    while True:
        try:
            # Hunt for volume opportunities
            result = await finance_minister.execute_task("Hunt for opportunities")
            
            if result.get("success") and result.get("metadata", {}).get("opportunities"):
                opps = result["metadata"]["opportunities"]
                console.print(f"\n[bold green]🎯 Volume spike detected: {len(opps)} opportunities[/bold green]\n")
        
        except Exception as e:
            console.print(f"[red]Volume loop error: {e}[/red]")
        
        # Check every 2 minutes
        await asyncio.sleep(120)


async def report_pnl_loop(finance_minister):
    """PnL reporting loop - every 5 minutes."""
    console.print("[yellow]📈 PnL reporting loop starting...[/yellow]\n")
    
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            
            # Load ledger
            from src.core.settings import Settings
            settings = Settings()
            
            try:
                with open(settings.paper_ledger_path, 'r') as f:
                    ledger = json.load(f)
                
                stats = ledger.get("stats", {})
                trades = ledger.get("trades", [])
                balances = ledger.get("balances", {})
                
                # Create report
                report_table = Table(title="📊 Paper Trading Report")
                report_table.add_column("Metric", style="cyan")
                report_table.add_column("Value", style="yellow")
                
                report_table.add_row("Total Trades", str(stats.get("total_trades", 0)))
                report_table.add_row("Profitable Trades", str(stats.get("profitable_trades", 0)))
                report_table.add_row("Total PnL", f"${stats.get('total_pnl', 0):.2f}")
                report_table.add_row("Spot Balance", f"${balances.get('spot_usdt', 0):.2f}")
                report_table.add_row("Futures Balance", f"${balances.get('futures_usdt', 0):.2f}")
                
                console.print("\n")
                console.print(report_table)
                
                # Show recent trades
                if trades:
                    recent = trades[-3:]  # Last 3 trades
                    console.print("\n[bold]Recent Trades:[/bold]")
                    for trade in recent:
                        console.print(f"  • {trade['market_type']} {trade['side']} {trade['symbol']} @ ${trade['entry_price']:.2f}")
                
                console.print("\n")
            
            except FileNotFoundError:
                console.print("[yellow]No trades yet[/yellow]")
        
        except Exception as e:
            console.print(f"[red]PnL report error: {e}[/red]")


async def start_hybrid_simulation():
    """Start the hybrid simulation system."""
    
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]🌐 HYBRID SIMULATION SYSTEM[/bold cyan]")
    console.print("[dim]News-Driven + Volume-Driven | Spot + Futures | Paper Trading[/dim]")
    console.print("=" * 70 + "\n")
    
    # Initialize ministers
    console.print("[yellow]Initializing ministers...[/yellow]")
    
    try:
        from src.government.ministers.finance_minister import FinanceMinister
        from src.government.ministers.foreign_minister import ForeignMinister
        from src.services.local_brain_service import LocalBrainService
        
        brain = LocalBrainService()
        
        finance = FinanceMinister(brain=brain)
        foreign = ForeignMinister(brain=brain)
        
        console.print("[green]✅ Ministers initialized[/green]\n")
        
        # Display config
        config_table = Table(title="System Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="yellow")
        
        config_table.add_row("Trading Mode", "PAPER (Simulation Only)")
        config_table.add_row("Spot Trading", "✅ Enabled")
        config_table.add_row("Futures Trading", "✅ Enabled")
        config_table.add_row("Starting Balance (Spot)", "$10,000")
        config_table.add_row("Starting Balance (Futures)", "$10,000")
        config_table.add_row("News Check Interval", "5 minutes")
        config_table.add_row("Volume Check Interval", "2 minutes")
        config_table.add_row("PnL Report Interval", "5 minutes")
        
        console.print(config_table)
        console.print()
        
        # Start loops
        console.print("[bold green]🚀 Hybrid Simulation Active![/bold green]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")
        
        # Run all loops concurrently
        await asyncio.gather(
            run_news_loop(finance, foreign),
            run_volume_loop(finance),
            report_pnl_loop(finance)
        )
    
    except Exception as e:
        console.print(f"[red]❌ Fatal error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    console.print("\n[bold cyan]Starting Hybrid Simulation System...[/bold cyan]")
    console.print("[yellow]⚠️ PAPER TRADING ONLY - NO REAL MONEY![/yellow]\n")
    
    try:
        asyncio.run(start_hybrid_simulation())
    except KeyboardInterrupt:
        console.print("\n\n[green]✅ Simulation stopped[/green]\n")
