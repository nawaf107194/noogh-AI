#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-End Brain Test - Meta-Llama-3-8B on RTX 5070
==================================================

Test the President with Meta-Llama-3-8B-Instruct neural core.
"""

import asyncio
import time
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


async def test_high_end_brain():
    """Test President with Meta-Llama-3-8B brain."""
    
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]🧠 Noogh Neural Core Test - Meta-Llama-3-8B-Instruct[/bold cyan]")
    console.print("=" * 70 + "\n")
    
    # Import President
    console.print("[yellow]Initializing President with Neural Core...[/yellow]")
    start_init = time.time()
    
    try:
        from src.government.president import President
        
        president = President(verbose=True)
        init_time = time.time() - start_init
        
        console.print(f"[green]✅ President initialized in {init_time:.2f}s[/green]\n")
        
        # Check if brain is available
        if not hasattr(president, 'brain') or president.brain is None:
            console.print("[red]❌ Neural Core not available![/red]")
            console.print("Run: python scripts/setup_local_model.py")
            console.print("Or install dependencies:")
            console.print("  pip install torch transformers accelerate sentencepiece")
            return
        
        # Get brain info
        from src.services.local_brain_service import LocalBrainService
        brain_info = LocalBrainService.get_model_info()
        
        console.print("[bold]Neural Core Information:[/bold]")
        console.print(f"   Status: [green]{brain_info.get('status', 'unknown')}[/green]")
        console.print(f"   Device: [cyan]{brain_info.get('device', 'unknown')}[/cyan]")
        console.print(f"   Parameters: [yellow]{brain_info.get('parameters_millions', 'unknown')}[/yellow]")
        console.print(f"   Model Type: [magenta]{brain_info.get('model_type', 'unknown')}[/magenta]\n")
        
    except Exception as e:
        console.print(f"[red]❌ Error initializing President: {e}[/red]")
        import traceback
        traceback.print_exc()
        return
    
    # Test queries
    test_queries = [
        "Explain the theory of relativity in simple terms.",
        "What is quantum computing and how does it work?",
        "Write a short poem about artificial intelligence.",
        "What are the main differences between supervised and unsupervised learning?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        console.print(f"\n[bold cyan]Test {i}/{len(test_queries)}:[/bold cyan]")
        console.print(f"[yellow]Question:[/yellow] {query}")
        
        start_time = time.time()
        
        try:
            # Process through President
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Thinking...", total=None)
                
                result = await president.process_request(
                    user_input=query,
                    context={},
                    priority="high"
                )
                
                progress.remove_task(task)
            
            inference_time = time.time() - start_time
            
            # Extract response
            if isinstance(result, dict):
                response = result.get('result', {}).get('message', str(result))
                minister = result.get('minister', 'unknown')
                status = result.get('status', 'unknown')
                
                # Display result
                console.print(Panel(
                    f"[bold]Response:[/bold]\n{response}\n\n"
                    f"[dim]Minister:[/dim] {minister}\n"
                    f"[dim]Status:[/dim] {status}\n"
                    f"[dim]Time:[/dim] {inference_time:.2f}s",
                    title="[green]🧠 Neural Core Response[/green]",
                    border_style="green"
                ))
                
                # Calculate tokens/second (rough estimate)
                estimated_tokens = len(response.split())
                if inference_time > 0:
                    tokens_per_sec = estimated_tokens / inference_time
                    console.print(f"[dim]   ~{tokens_per_sec:.1f} words/second[/dim]")
            else:
                console.print(f"[yellow]Result: {result}[/yellow]")
        
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            import traceback
            traceback.print_exc()
    
    console.print("\n" + "=" * 70)
    console.print("[bold green]✨ Neural Core test complete![/bold green]")
    console.print("=" * 70 + "\n")


async def test_direct_brain():
    """Test the brain directly (without President overhead)."""
    
    console.print("\n[bold cyan]🔬 Direct Neural Core Test[/bold cyan]\n")
    
    try:
        from src.services.local_brain_service import LocalBrainService
        
        brain = LocalBrainService()
        
        prompt = "Explain why the sky is blue in one paragraph."
        console.print(f"[yellow]Prompt:[/yellow] {prompt}\n")
        
        start_time = time.time()
        response = await brain.think(prompt, max_tokens=256)
        inference_time = time.time() - start_time
        
        console.print(Panel(
            response,
            title=f"[green]Response ({inference_time:.2f}s)[/green]",
            border_style="cyan"
        ))
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "direct":
        asyncio.run(test_direct_brain())
    else:
        asyncio.run(test_high_end_brain())
