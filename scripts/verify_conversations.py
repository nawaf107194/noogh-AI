#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversation Verification Script
=================================

Verify that conversations are being saved to the database.
"""

import asyncio
import httpx
from rich.console import Console
from rich.table import Table
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from src.core.settings import settings
from src.core.database import get_async_database_url
from src.models.conversation import Conversation

console = Console()

BASE_URL = "http://localhost:8000"


async def test_conversation_persistence():
    """Test that conversations are saved to database."""
    
    console.print("\n[bold cyan]💾 Testing Conversation Persistence[/bold cyan]\n")
    
    # Setup database connection
    db_url = get_async_database_url(settings.database_url)
    engine = create_async_engine(db_url)
    async_session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        
        # Test 1: Send a message to the President
        console.print("[yellow]1. Sending message to President...[/yellow]")
        test_message = "What is the meaning of life?"
        
        try:
            response = await client.post(
                "/api/v1/government/chat",
                json={
                    "message": test_message,
                    "priority": "medium"
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                console.print(f"[green]✅ President responded![/green]")
                console.print(f"   Response: {data.get('response', 'No response')[:100]}...")
                console.print(f"   Minister: {data.get('minister')}")
                console.print(f"   Intent: {data.get('intent')}")
            else:
                console.print(f"[red]❌ Request failed: {response.status_code}[/red]")
                return
        
        except httpx.ConnectError:
            console.print("[bold red]❌ Cannot connect to server![/bold red]")
            console.print("Make sure the server is running:")
            console.print("  python -m src.api.main")
            return
    
    # Test 2: Query database to verify conversation was saved
    console.print("\n[yellow]2. Checking database for saved conversation...[/yellow]")
    
    async with async_session_factory() as db:
        # Count total conversations
        count_stmt = select(func.count(Conversation.id))
        result = await db.execute(count_stmt)
        total_count = result.scalar()
        
        console.print(f"[green]✅ Total conversations in database: {total_count}[/green]")
        
        # Get recent conversations
        stmt = select(Conversation).order_by(Conversation.created_at.desc()).limit(5)
        result = await db.execute(stmt)
        conversations = list(result.scalars().all())
        
        if conversations:
            console.print(f"\n[bold]Recent Conversations:[/bold]")
            
            table = Table(title="Conversation History")
            table.add_column("ID", style="cyan")
            table.add_column("User Input", style="yellow")
            table.add_column("Minister", style="green")
            table.add_column("Intent", style="magenta")
            table.add_column("Status", style="blue")
            table.add_column("Time (ms)", style="white")
            
            for conv in conversations:
                table.add_row(
                    str(conv.id),
                    conv.user_input[:50] + "..." if len(conv.user_input) > 50 else conv.user_input,
                    conv.minister_name or "N/A",
                    conv.intent or "N/A",
                    conv.status or "N/A",
                    f"{conv.execution_time_ms:.1f}" if conv.execution_time_ms else "N/A"
                )
            
            console.print(table)
            
            # Check if our test message is in there
            if any(test_message in conv.user_input for conv in conversations):
                console.print("\n[green]✅ Test conversation successfully saved![/green]")
            else:
                console.print("\n[yellow]⚠️  Test conversation not found in recent history[/yellow]")
        else:
            console.print("[red]❌ No conversations found in database[/red]")
    
    await engine.dispose()
    console.print("\n[bold green]✨ Verification complete![/bold green]\n")


async def show_conversation_stats():
    """Show statistics about saved conversations."""
    
    console.print("\n[bold cyan]📊 Conversation Statistics[/bold cyan]\n")
    
    # Setup database connection
    db_url = get_async_database_url(settings.database_url)
    engine = create_async_engine(db_url)
    async_session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    
    async with async_session_factory() as db:
        # Total count
        count_stmt = select(func.count(Conversation.id))
        result = await db.execute(count_stmt)
        total = result.scalar()
        
        # Count by status
        completed_stmt = select(func.count(Conversation.id)).where(Conversation.status == "completed")
        result = await db.execute(completed_stmt)
        completed = result.scalar()
        
        # Count by minister
        from sqlalchemy import distinct
        ministers_stmt = select(func.count(distinct(Conversation.minister_name)))
        result = await db.execute(ministers_stmt)
        unique_ministers = result.scalar()
        
        console.print(f"Total Conversations: [green]{total}[/green]")
        console.print(f"Completed: [green]{completed}[/green]")
        console.print(f"Unique Ministers: [cyan]{unique_ministers}[/cyan]")
        
        if total > 0:
            # Average execution time
            avg_stmt = select(func.avg(Conversation.execution_time_ms))
            result = await db.execute(avg_stmt)
            avg_time = result.scalar()
            console.print(f"Average Execution Time: [yellow]{avg_time:.2f}ms[/yellow]")
    
    await engine.dispose()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        asyncio.run(show_conversation_stats())
    else:
        asyncio.run(test_conversation_persistence())
