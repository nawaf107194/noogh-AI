#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autonomous Hunter - The Silent Market Scanner
==============================================

Runs 24/7 scanning for high-probability trading setups.
Uses multi-modal AI (Vision + LLM) for confirmation.

SAFETY FIRST: Monitors GPU temperature to protect RTX 5070.
"""

import asyncio
import time
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table

console = Console()


async def check_system_health():
    """Check if system is healthy enough to trade."""
    try:
        from src.government.ministers.health_minister import HealthMinister
        from src.services.local_brain_service import LocalBrainService
        
        brain = LocalBrainService()
        health = HealthMinister(brain=brain)
        
        vitals = health.check_vital_signs()
        gpu = vitals.get("gpu", {})
        
        temp = gpu.get("temperature_c", 0)
        
        # SAFETY: Don't trade if GPU is too hot
        if temp > 80:
            return {
                "healthy": False,
                "reason": f"GPU temperature too high: {temp}°C",
                "temp": temp
            }
        
        return {
            "healthy": True,
            "temp": temp,
            "vram": gpu.get("vram_percent", 0)
        }
    
    except Exception as e:
        console.print(f"[red]❌ Health check error: {e}[/red]")
        return {"healthy": False, "reason": str(e)}


async def hunt_opportunities():
    """Execute the hunt."""
    try:
        from src.government.ministers.finance_minister import FinanceMinister
        from src.services.local_brain_service import LocalBrainService
        
        brain = LocalBrainService()
        finance = FinanceMinister(brain=brain)
        
        result = await finance.execute_task("Hunt for opportunities")
        
        return result
    
    except Exception as e:
        console.print(f"[red]❌ Hunt error: {e}[/red]")
        return {"success": False, "error": str(e)}


async def alert_user(opportunities):
    """Alert user of opportunities."""
    try:
        from src.government.ministers.communication_minister import CommunicationMinister
        from src.services.local_brain_service import LocalBrainService
        
        brain = LocalBrainService()
        comm = CommunicationMinister(brain=brain)
        
        for opp in opportunities:
            symbol = opp.get("symbol", "UNKNOWN")
            price = opp.get("price", 0)
            rvol = opp.get("rvol", 0)
            
            await comm.execute_task(
                f"BUY SIGNAL for {symbol} at ${price:.2f} (RVOL: {rvol:.2f}x)",
                context={"format": "tweet"}
            )
    
    except Exception as e:
        console.print(f"[yellow]⚠️ Alert error: {e}[/yellow]")


async def run_hunter_loop(interval: int = 60):
    """
    The Silent Hunter - Infinite loop.
    
    Args:
        interval: Seconds between scans
    """
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]🎯 AUTONOMOUS HUNTER ACTIVATED[/bold cyan]")
    console.print("[dim]Multi-Modal AI Trading System | Volume Spike Detection | 24/7[/dim]")
    console.print("=" * 70 + "\n")
    
    cycle = 0
    total_signals = 0
    
    try:
        while True:
            cycle += 1
            
            # Status table
            status_table = Table(title=f"Hunter Cycle #{cycle}")
            status_table.add_column("Metric", style="cyan")
            status_table.add_column("Value", style="yellow")
            
            status_table.add_row("Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            status_table.add_row("Total Signals Found", str(total_signals))
            status_table.add_row("Status", "🏹 Hunting...")
            
            console.print(status_table)
            
            # Step 1: Health Check
            console.print("\n[yellow]🏥 Checking system health...[/yellow]")
            health = await check_system_health()
            
            if not health.get("healthy"):
                console.print(f"[red]⚠️ TRADING PAUSED: {health.get('reason')}[/red]")
                console.print(f"[dim]Waiting {interval}s before retry...[/dim]\n")
                await asyncio.sleep(interval)
                continue
            
            console.print(f"[green]✅ System healthy (GPU: {health.get('temp')}°C)[/green]")
            
            # Step 2: Hunt
            console.print("\n[yellow]🎯 Scanning markets for volume spikes...[/yellow]")
            hunt_result = await hunt_opportunities()
            
            if hunt_result.get("success"):
                metadata = hunt_result.get("metadata", {})
                opportunities = metadata.get("opportunities", [])
                scanned = metadata.get("scanned", 0)
                
                console.print(f"[cyan]📊 Scanned {scanned} pairs[/cyan]")
                
                if opportunities:
                    console.print(f"\n[bold green]🚨 {len(opportunities)} BUY SIGNALS DETECTED! 🚨[/bold green]\n")
                    
                    for opp in opportunities:
                        console.print(Panel(
                            f"[bold]Symbol:[/bold] {opp['symbol']}\n"
                            f"[bold]Price:[/bold] ${opp['price']:.2f}\n"
                            f"[bold]RVOL:[/bold] {opp['rvol']:.2f}x (Volume Spike!)\n"
                            f"[bold]RSI:[/bold] {opp.get('rsi', 'N/A')}\n"
                            f"[bold]Vision AI:[/bold] {'✅ Bullish' if opp.get('vision_bullish') else '❌ Neutral'}\n"
                            f"[bold]LLM AI:[/bold] {'✅ Bullish' if opp.get('llm_bullish') else '❌ Neutral'}",
                            title=f"[green]🔥 BUY SIGNAL: {opp['symbol']}[/green]",
                            border_style="green"
                        ))
                    
                    total_signals += len(opportunities)
                    
                    # Alert user
                    await alert_user(opportunities)
                else:
                    console.print("[dim]No signals this cycle. Market is quiet.[/dim]")
            else:
                console.print(f"[red]❌ Hunt failed: {hunt_result.get('error')}[/red]")
            
            # Sleep
            console.print(f"\n[dim]💤 Next scan in {interval} seconds...[/dim]\n")
            console.print("─" * 70 + "\n")
            
            await asyncio.sleep(interval)
    
    except KeyboardInterrupt:
        console.print("\n\n[yellow]🛑 Hunter stopped by user[/yellow]")
        console.print(f"[green]✅ Total signals found: {total_signals}[/green]\n")
    except Exception as e:
        console.print(f"\n\n[red]❌ Fatal error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    console.print("\n[bold cyan]Starting Autonomous Hunter...[/bold cyan]")
    console.print("[yellow]⚠️ DISCLAIMER: This is for educational purposes only![/yellow]")
    console.print("[yellow]⚠️ NOT financial advice. Trade at your own risk![/yellow]\n")
    
    try:
        asyncio.run(run_hunter_loop(interval=60))
    except KeyboardInterrupt:
        console.print("\n[green]Shutdown complete[/green]\n")
