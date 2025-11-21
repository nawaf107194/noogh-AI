#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Verification Script - Test User Domain
==========================================

Script to test the complete User Domain vertical slice.
"""

import asyncio
import httpx
from rich.console import Console
from rich.table import Table

console = Console()

BASE_URL = "http://localhost:8000"


async def test_user_api():
    """Test the User API endpoints."""
    
    console.print("\n[bold cyan]🧪 Testing User Domain API[/bold cyan]\n")
    
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        
        # Test 1: Create User
        console.print("[yellow]1. Creating a new user...[/yellow]")
        create_response = await client.post(
            "/api/v1/users/",
            json={
                "email": "john.doe@example.com",
                "username": "johndoe",
                "full_name": "John Doe",
                "password": "SecurePassword123!",
                "is_active": True,
                "is_superuser": False
            }
        )
        
        if create_response.status_code == 201:
            user_data = create_response.json()
            user_id = user_data["id"]
            console.print(f"[green]✅ User created successfully! ID: {user_id}[/green]")
            console.print(f"   Email: {user_data['email']}")
            console.print(f"   Username: {user_data['username']}")
            console.print(f"   Created: {user_data['created_at']}")
        else:
            console.print(f"[red]❌ Failed to create user: {create_response.status_code}[/red]")
            console.print(create_response.text)
            return
        
        # Test 2: Get User by ID
        console.print(f"\n[yellow]2. Getting user by ID ({user_id})...[/yellow]")
        get_response = await client.get(f"/api/v1/users/{user_id}")
        
        if get_response.status_code == 200:
            console.print("[green]✅ User retrieved successfully![/green]")
        else:
            console.print(f"[red]❌ Failed to get user: {get_response.status_code}[/red]")
        
        # Test 3: List Users
        console.print("\n[yellow]3. Listing all users...[/yellow]")
        list_response = await client.get("/api/v1/users/?skip=0&limit=10")
        
        if list_response.status_code == 200:
            data = list_response.json()
            console.print(f"[green]✅ Found {data['total']} users[/green]")
            
            # Display users in table
            table = Table(title="Users")
            table.add_column("ID", style="cyan")
            table.add_column("Email", style="green")
            table.add_column("Username", style="yellow")
            table.add_column("Active", style="magenta")
            
            for user in data["users"]:
                table.add_row(
                    str(user["id"]),
                    user["email"],
                    user["username"],
                    "✓" if user["is_active"] else "✗"
                )
            
            console.print(table)
        else:
            console.print(f"[red]❌ Failed to list users: {list_response.status_code}[/red]")
        
        # Test 4: Update User
        console.print(f"\n[yellow]4. Updating user {user_id}...[/yellow]")
        update_response = await client.put(
            f"/api/v1/users/{user_id}",
            json={
                "full_name": "John Updated Doe"
            }
        )
        
        if update_response.status_code == 200:
            updated = update_response.json()
            console.print(f"[green]✅ User updated! New name: {updated['full_name']}[/green]")
        else:
            console.print(f"[red]❌ Failed to update user: {update_response.status_code}[/red]")
        
        # Test 5: Deactivate User
        console.print(f"\n[yellow]5. Deactivating user {user_id}...[/yellow]")
        deactivate_response = await client.post(f"/api/v1/users/{user_id}/deactivate")
        
        if deactivate_response.status_code == 200:
            console.print("[green]✅ User deactivated successfully![/green]")
        else:
            console.print(f"[red]❌ Failed to deactivate: {deactivate_response.status_code}[/red]")
        
        # Test 6: Reactivate User
        console.print(f"\n[yellow]6. Reactivating user {user_id}...[/yellow]")
        activate_response = await client.post(f"/api/v1/users/{user_id}/activate")
        
        if activate_response.status_code == 200:
            console.print("[green]✅ User reactivated successfully![/green]")
        else:
            console.print(f"[red]❌ Failed to reactivate: {activate_response.status_code}[/red]")
        
        # Test 7: Try to create duplicate (should fail)
        console.print("\n[yellow]7. Testing duplicate email prevention...[/yellow]")
        duplicate_response = await client.post(
            "/api/v1/users/",
            json={
                "email": "john.doe@example.com",  # Same email
                "username": "johndoe2",
                "full_name": "John Duplicate",
                "password": "Password123!"
            }
        )
        
        if duplicate_response.status_code == 400:
            console.print("[green]✅ Duplicate email correctly rejected![/green]")
            console.print(f"   Error: {duplicate_response.json()['detail']}")
        else:
            console.print(f"[red]❌ Should have rejected duplicate email![/red]")
        
        console.print("\n[bold green]✨ All tests completed![/bold green]\n")


if __name__ == "__main__":
    try:
        asyncio.run(test_user_api())
    except httpx.ConnectError:
        console.print("[bold red]❌ Could not connect to API server![/bold red]")
        console.print("Make sure the server is running:")
        console.print("  python -m src.api.main")
