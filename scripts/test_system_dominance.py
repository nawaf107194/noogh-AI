#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Dominance Test - Ultimate Integration Verification
===========================================================

Tests complete system control:
1. Hardware monitoring (GPU/CPU/Devices)
2. File system indexing
3. Resource optimization with AI recommendations
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


async def test_hardware_dominance():
    """Test 1: Hardware monitoring and peripheral detection."""
    console.print("\n[bold cyan]═══ Test 1: Hardware Dominance ═══[/bold cyan]\n")
    
    try:
        from src.government.ministers.health_minister import HealthMinister
        from src.services.local_brain_service import LocalBrainService
        
        brain = LocalBrainService()
        health = HealthMinister(brain=brain)
        
        # Get vital signs
        console.print("[yellow]📊 Checking system vital signs...[/yellow]\n")
        vitals = health.check_vital_signs()
        
        # Display hardware stats
        gpu = vitals.get("gpu", {})
        cpu = vitals.get("cpu", {})
        mem = vitals.get("memory", {})
        disk = vitals.get("disk", {})
        
        hardware_table = Table(title="System Hardware Status")
        hardware_table.add_column("Component", style="cyan")
        hardware_table.add_column("Metric", style="yellow")
        hardware_table.add_column("Value", style="green")
        
        hardware_table.add_row("GPU", "Name", gpu.get("name", "N/A"))
        hardware_table.add_row("GPU", "Temperature", f"{gpu.get('temperature_c', 'N/A')}°C")
        hardware_table.add_row("GPU", "VRAM Usage", f"{gpu.get('vram_percent', 0):.1f}%")
        hardware_table.add_row("CPU", "Usage", f"{cpu.get('percent', 'N/A')}%")
        hardware_table.add_row("CPU", "Cores", str(cpu.get('cores', 'N/A')))
        hardware_table.add_row("RAM", "Usage", f"{mem.get('percent', 'N/A')}%")
        hardware_table.add_row("Disk", "Usage", f"{disk.get('percent', 'N/A')}%")
        
        console.print(hardware_table)
        
        # Monitor peripherals
        console.print("\n[yellow]🔌 Scanning peripherals...[/yellow]\n")
        peripheral_result = health.monitor_peripherals()
        
        if peripheral_result.get("success"):
            devices = peripheral_result.get("devices", [])
            console.print(f"[green]✅ Detected {len(devices)} peripheral devices[/green]\n")
            
            if devices:
                device_table = Table(title="Connected Devices")
                device_table.add_column("Name", style="cyan")
                device_table.add_column("Type", style="yellow")
                device_table.add_column("Details", style="dim")
                
                for device in devices[:5]:  # Show first 5
                    device_table.add_row(
                        device.get("name", "Unknown"),
                        device.get("type", "unknown"),
                        device.get("model", device.get("vendor", "N/A"))
                    )
                
                console.print(device_table)
        else:
            console.print(f"[yellow]⚠️ Peripheral detection: {peripheral_result.get('error')}[/yellow]")
        
        return vitals
    
    except Exception as e:
        console.print(f"[red]❌ Hardware test failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


async def test_file_dominance():
    """Test 2: File system indexing."""
    console.print("\n[bold cyan]═══ Test 2: File System Dominance ═══[/bold cyan]\n")
    
    try:
        from src.government.ministers.development_minister import DevelopmentMinister
        from src.services.local_brain_service import LocalBrainService
        
        brain = LocalBrainService()
        development = DevelopmentMinister(brain=brain)
        
        console.print("[yellow]📂 Indexing project directory...[/yellow]\n")
        
        # Index current project (max depth 2 to avoid overwhelming)
        context = {"path": ".", "max_depth": 2}
        result = await development.execute_task("Index this directory", context=context)
        
        if result.get("success"):
            metadata = result.get("metadata", {})
            index_data = metadata.get("index_data", {})
            
            total_files = index_data.get("total_files", 0)
            total_size = index_data.get("total_size_mb", 0)
            file_types = index_data.get("file_types", {})
            
            # Display index stats
            console.print(Panel(
                f"[bold]Files Indexed:[/bold] {total_files}\n"
                f"[bold]Total Size:[/bold] {total_size:.2f} MB\n"
                f"[bold]Python Files:[/bold] {file_types.get('.py', 0)}\n"
                f"[bold]Markdown Files:[/bold] {file_types.get('.md', 0)}\n"
                f"[bold]JSON Files:[/bold] {file_types.get('.json', 0)}",
                title="[green]File Index Summary[/green]",
                border_style="green"
            ))
            
            # Show AI analysis
            console.print("\n[bold]AI Storage Recommendations:[/bold]")
            console.print(Panel(
                result['response'],
                title="[cyan]File Master Analysis[/cyan]",
                border_style="cyan"
            ))
            
            return index_data
        else:
            console.print(f"[red]❌ Indexing failed: {result.get('error')}[/red]")
            return None
    
    except Exception as e:
        console.print(f"[red]❌ File indexing test failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


async def test_resource_optimization():
    """Test 3: Resource optimization with AI."""
    console.print("\n[bold cyan]═══ Test 3: Resource Optimization ═══[/bold cyan]\n")
    
    try:
        from src.government.ministers.health_minister import HealthMinister
        from src.services.local_brain_service import LocalBrainService
        
        brain = LocalBrainService()
        health = HealthMinister(brain=brain)
        
        console.print("[yellow]⚡ Analyzing resource usage...[/yellow]\n")
        
        result = await health.execute_task("Optimize system resources")
        
        if result.get("success"):
            metadata = result.get("metadata", {})
            top_procs = metadata.get("top_processes", [])[:3]
            
            # Display top processes
            if top_procs:
                proc_table = Table(title="Top 3 Memory Consumers")
                proc_table.add_column("Process", style="cyan")
                proc_table.add_column("RAM %", justify="right", style="yellow")
                proc_table.add_column("CPU %", justify="right", style="yellow")
                
                for proc in top_procs:
                    proc_table.add_row(
                        proc.get("name", "Unknown"),
                        f"{proc.get('memory_percent', 0):.1f}%",
                        f"{proc.get('cpu_percent', 0):.1f}%"
                    )
                
                console.print(proc_table)
            
            # Show AI recommendations
            console.print("\n[bold]AI Optimization Recommendations:[/bold]")
            console.print(Panel(
                result['response'],
                title="[red]Resource Warlord Analysis[/red]",
                border_style="red"
            ))
            
            return top_procs
        else:
            console.print(f"[red]❌ Optimization failed: {result.get('error')}[/red]")
            return None
    
    except Exception as e:
        console.print(f"[red]❌ Optimization test failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Run all system dominance tests."""
    
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]🖥️ SYSTEM DOMINANCE TEST - PHASE 14[/bold cyan]")
    console.print("[dim]Total OS Integration & Control[/dim]")
    console.print("=" * 70)
    
    # Test 1: Hardware
    hardware = await test_hardware_dominance()
    
    # Test 2: Files
    files = await test_file_dominance()
    
    # Test 3: Optimization
    resources = await test_resource_optimization()
    
    # Summary
    console.print("\n" + "=" * 70)
    console.print("[bold green]✨ SYSTEM DOMINANCE TEST COMPLETE[/bold green]")
    console.print("=" * 70)
    
    summary_table = Table(title="Test Results")
    summary_table.add_column("Test", style="cyan")
    summary_table.add_column("Status", style="bold")
    summary_table.add_column("Data Points", justify="right")
    
    summary_table.add_row(
        "Hardware Monitoring",
        "✅ Operational" if hardware else "❌ Failed",
        "GPU/CPU/RAM/Devices"
    )
    summary_table.add_row(
        "File Indexing",
        "✅ Operational" if files else "❌ Failed",
        f"{files.get('total_files', 0)} files" if files else "0"
    )
    summary_table.add_row(
        "Resource Optimization",
        "✅ Operational" if resources else "❌ Failed",
        f"{len(resources)} processes" if resources else "0"
    )
    
    console.print(summary_table)
    
    console.print("\n[bold]🎉 THE SYSTEM HAS ACHIEVED FULL DOMINANCE![/bold]")
    console.print("[dim]Hardware Control | File System Mastery | Resource Optimization | AI-Powered Decisions[/dim]\n")


if __name__ == "__main__":
    asyncio.run(main())
