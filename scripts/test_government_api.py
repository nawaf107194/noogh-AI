#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Government API Integration Test Script
=======================================

Test the modernized Government API (President & Cabinet).
"""

import asyncio
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

BASE_URL = "http://localhost:8000"


async def test_government_api():
    """Test the Government API endpoints."""
    
    console.print("\n[bold cyan]🏛️ Testing Government API V2[/bold cyan]\n")
    
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        
        # Test 1: Health Check
        console.print("[yellow]1. Checking Government System health...[/yellow]")
        try:
            health_response = await client.get("/api/v1/government/health")
            
            if health_response.status_code == 200:
                health_data = health_response.json()
                console.print(f"[green]✅ Government System is {health_data['status']}[/green]")
                console.print(f"   President initialized: {health_data.get('president_initialized', False)}")
                console.print(f"   Ministers: {health_data.get('ministers_count', 0)}")
            else:
                console.print(f"[red]❌ Health check failed: {health_response.status_code}[/red]")
        except httpx.ConnectError:
            console.print("[bold red]❌ Cannot connect to server![/bold red]")
            console.print("Make sure the server is running:")
            console.print("  python -m src.api.main")
            return
        
        # Test 2: Get Cabinet Status
        console.print("\n[yellow]2. Getting Cabinet status...[/yellow]")
        status_response = await client.get("/api/v1/government/status")
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            console.print("[green]✅ Cabinet Status Retrieved[/green]")
            
            # Display status in table
            table = Table(title="Cabinet Status")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Ministers", str(status_data.get("total_ministers", 0)))
            table.add_row("Active Ministers", str(status_data.get("active_ministers", 0)))
            table.add_row("Total Requests", str(status_data.get("total_requests", 0)))
            table.add_row("Successful Requests", str(status_data.get("successful_requests", 0)))
            table.add_row("Success Rate", f"{status_data.get('success_rate', 0):.1%}")
            
            console.print(table)
            console.print(f"   Ministers: {', '.join(status_data.get('ministers', []))}")
        else:
            console.print(f"[red]❌ Failed to get status: {status_response.status_code}[/red]")
        
        # Test 3: Chat with President - System Status
        console.print("\n[yellow]3. Asking President about system status...[/yellow]")
        chat_response_1 = await client.post(
            "/api/v1/government/chat",
            json={
                "message": "What is the system status?",
                "priority": "high"
            }
        )
        
        if chat_response_1.status_code == 200:
            chat_data_1 = chat_response_1.json()
            console.print(Panel(
                f"[bold]Response:[/bold] {chat_data_1.get('response', 'No response')}\n"
                f"[dim]Minister:[/dim] {chat_data_1.get('minister', 'Unknown')}\n"
                f"[dim]Intent:[/dim] {chat_data_1.get('intent', 'Unknown')}\n"
                f"[dim]Task ID:[/dim] {chat_data_1.get('task_id', 'N/A')}\n"
                f"[dim]Status:[/dim] {chat_data_1.get('status', 'Unknown')}",
                title="[green]President's Response[/green]",
                border_style="green"
            ))
        else:
            console.print(f"[red]❌ Chat failed: {chat_response_1.status_code}[/red]")
            console.print(chat_response_1. text)
        
        # Test 4: Chat with President - Educational Question
        console.print("\n[yellow]4. Asking President an educational question...[/yellow]")
        chat_response_2 = await client.post(
            "/api/v1/government/chat",
            json={
                "message": "Explain what is artificial intelligence",
                "priority": "medium"
            }
        )
        
        if chat_response_2.status_code == 200:
            chat_data_2 = chat_response_2.json()
            console.print(Panel(
                f"[bold]Response:[/bold] {chat_data_2.get('response', 'No response')}\n"
                f"[dim]Minister:[/dim] {chat_data_2.get('minister', 'Unknown')}\n"
                f"[dim]Intent:[/dim] {chat_data_2.get('intent', 'Unknown')}",
                title="[green]President's Response[/green]",
                border_style="green"
            ))
        else:
            console.print(f"[red]❌ Chat failed: {chat_response_2.status_code}[/red]")
        
        # Test 5: Chat with context
        console.print("\n[yellow]5. Asking with conversation context...[/yellow]")
        chat_response_3 = await client.post(
            "/api/v1/government/chat",
            json={
                "message": "Tell me more about that",
                "context": {
                    "history": ["What is artificial intelligence?"]
                },
                "priority": "low"
            }
        )
        
        if chat_response_3.status_code == 200:
            chat_data_3 = chat_response_3.json()
            console.print(Panel(
                f"[bold]Response:[/bold] {chat_data_3.get('response', 'No response')}\n"
                f"[dim]Minister:[/dim] {chat_data_3.get('minister', 'Unknown')}",
                title="[green]President's Response[/green]",
                border_style="green"
            ))
        
        console.print("\n[bold green]✨ All Government API tests completed![/bold green]\n")


if __name__ == "__main__":
    try:
        asyncio.run(test_government_api())
    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted by user[/yellow]")
