#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autonomous System Test - Multi-Agent Coordination
==================================================

Tests autonomous ministers working together:
1. Finance Minister analyzes BTC market
2. Education Minister creates trading strategy
3. Security Minister scans for threats
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


async def test_finance_minister():
    """Test Finance Minister - Autonomous Crypto Trading."""
    console.print("\n[bold cyan]═══ Test 1: Finance Minister (Crypto Trading) ═══[/bold cyan]\n")
    
    try:
        from src.government.ministers.finance_minister import FinanceMinister
        from src.services.local_brain_service import LocalBrainService
        
        # Initialize
        brain = LocalBrainService()
        finance = FinanceMinister(brain=brain)
        
        console.print("[yellow]🔍 Analyzing BTC/USDT market...[/yellow]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Fetching market data & calculating indicators...", total=None)
            
            result = await finance.execute_task("Analyze BTC/USDT market")
            
            progress.remove_task(task)
        
        if result.get("success"):
            metadata = result.get("metadata", {})
            analysis_data = metadata.get("analysis_data", {})
            
            # Display market data
            console.print(Panel(
                f"[bold]Symbol:[/bold] {metadata.get('symbol', 'N/A')}\n"
                f"[bold]Current Price:[/bold] ${analysis_data.get('current_price', 0):,.2f}\n"
                f"[bold]RSI:[/bold] {analysis_data.get('rsi', 'N/A'):.2f if analysis_data.get('rsi') else 'N/A'}\n"
                f"[bold]EMA 20:[/bold] ${analysis_data.get('ema_20', 0):,.2f}\n"
                f"[bold]24h Volume:[/bold] ${analysis_data.get('volume_24h', 0):,.0f}",
                title="[green]Market Data[/green]",
                border_style="green"
            ))
            
            # Display AI analysis
            console.print("\n[bold]AI Market Analysis:[/bold]")
            console.print(Panel(
                result['response'],
                title="[cyan]Meta-Llama-3-8B Trading Recommendations[/cyan]",
                border_style="cyan"
            ))
            
            return result
        else:
            console.print(f"[red]❌ Failed: {result.get('error')}[/red]")
            return None
    
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


async def test_education_minister(finance_result):
    """Test Education Minister - Strategy Generation."""
    console.print("\n[bold cyan]═══ Test 2: Education Minister (Strategy Optimizer) ═══[/bold cyan]\n")
    
    if not finance_result:
        console.print("[yellow]⚠️ Skipping (no finance data)[/yellow]\n")
        return None
    
    try:
        from src.government.ministers.education_minister import EducationMinister
        from src.services.local_brain_service import LocalBrainService
        
        # Initialize
        brain = LocalBrainService()
        education = EducationMinister(brain=brain)
        
        console.print("[yellow]🧠 Generating trading strategy...[/yellow]\n")
        
        # Prepare context with finance data
        context = {
            "minister_data": finance_result.get("metadata", {}),
            "data_type": "market"
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Strategizing...", total=None)
            
            result = await education.execute_task(
                "Create a trading strategy based on market analysis",
                context=context
            )
            
            progress.remove_task(task)
        
        if result.get("success"):
            console.print(Panel(
                result['response'],
                title="[green]AI-Generated Trading Strategy[/green]",
                border_style="green"
            ))
            return result
        else:
            console.print(f"[red]❌ Failed: {result.get('error')}[/red]")
            return None
    
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        return None


async def test_security_minister():
    """Test Security Minister - Threat Detection."""
    console.print("\n[bold cyan]═══ Test 3: Security Minister (Threat Scanner) ═══[/bold cyan]\n")
    
    try:
        from src.government.ministers.security_minister import SecurityMinister
        from src.services.local_brain_service import LocalBrainService
        
        # Initialize
        brain = LocalBrainService()
        security = SecurityMinister(brain=brain)
        
        # Simulate malicious logs
        fake_logs = [
            "GET /api/users?id=1 HTTP/1.1",
            "POST /login username=admin password=secure123",
            "GET /api/data?query=SELECT * FROM users WHERE id=1 OR '1'='1'",  # SQL injection!
            "GET /profile?name=<script>alert('XSS')</script>",  # XSS!
            "GET /files?path=../../etc/passwd",  # Path traversal!
        ]
        
        console.print("[yellow]🛡️ Scanning logs for threats...[/yellow]")
        console.print(f"[dim]Scanning {len(fake_logs)} log entries...[/dim]\n")
        
        context = {"logs": fake_logs}
        
        result = await security.execute_task(
            "Scan and mitigate threats",
            context=context
        )
        
        if result.get("success"):
            metadata = result.get("metadata", {})
            scan_result = metadata.get("scan_result", {})
            
            # Display scan summary
            console.print(Panel(
                f"[bold]Logs Scanned:[/bold] {scan_result.get('total_logs_scanned', 0)}\n"
                f"[bold]Threats Found:[/bold] {scan_result.get('threat_count', 0)}\n",
                title="[yellow]Scan Results[/yellow]",
                border_style="yellow"
            ))
            
            # Display mitigation
            console.print("\n[bold]AI-Generated Mitigation:[/bold]")
            console.print(Panel(
                result['response'],
                title="[red]Security Patch & Recommendations[/red]",
                border_style="red"
            ))
            
            return result
        else:
            console.print(f"[red]❌ Failed: {result.get('error')}[/red]")
            return None
    
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        return None


async def main():
    """Run all autonomous system tests."""
    
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]🤖 AUTONOMOUS AGENT SYSTEM TEST[/bold cyan]")
    console.print("=" * 70)
    
    # Test Finance → Education → Security pipeline
    finance_result = await test_finance_minister()
    await asyncio.sleep(1)
    
    education_result = await test_education_minister(finance_result)
    await asyncio.sleep(1)
    
    security_result = await test_security_minister()
    
    # Summary
    console.print("\n" + "=" * 70)
    console.print("[bold green]✨ AUTONOMOUS SYSTEM TEST COMPLETE[/bold green]")
    console.print("=" * 70)
    
    results_table = Table(title="Test Results")
    results_table.add_column("Minister", style="cyan")
    results_table.add_column("Status", style="bold")
    results_table.add_column("Capability", style="yellow")
    
    results_table.add_row(
        "Finance",
        "✅ Operational" if finance_result else "❌ Failed",
        "Crypto trading, technical analysis"
    )
    results_table.add_row(
        "Education",
        "✅ Operational" if education_result else "❌ Failed",
        "Strategy generation, optimization"
    )
    results_table.add_row(
        "Security",
        "✅ Operational" if security_result else "❌ Failed",
        "Threat detection, patch generation"
    )
    
    console.print(results_table)
    console.print("\n[bold]🎉 The Autonomous Agent Government is LIVE![/bold]\n")


if __name__ == "__main__":
    asyncio.run(main())
