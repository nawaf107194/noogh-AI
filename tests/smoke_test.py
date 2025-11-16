#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import asyncio
from datetime import datetime
import httpx

BASE_URL = os.getenv("NOOGH_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("NOOGH_TEST_API_KEY", "dev-test-key")

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}


async def wait_for_server(timeout=30):
    start = datetime.now()
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=5) as client:
        while (datetime.now() - start).seconds < timeout:
            try:
                r = await client.get("/health")
                if r.status_code == 200:
                    return True
            except Exception:
                await asyncio.sleep(1)
        return False


async def get(client, path, headers=None):
    r = await client.get(path, headers=headers)
    return r


async def post(client, path, json=None, headers=None):
    r = await client.post(path, json=json, headers=headers)
    return r


async def put(client, path, json=None, headers=None):
    r = await client.put(path, json=json, headers=headers)
    return r


async def delete(client, path, headers=None):
    r = await client.delete(path, headers=headers)
    return r


def assert_ok(resp, expected=200):
    if resp.status_code != expected:
        print(f"FAIL {resp.request.method} {resp.request.url} -> {resp.status_code} != {expected}")
        print(resp.text)
        sys.exit(1)


async def run_tests():
    server_ready = await wait_for_server(timeout=40)
    if not server_ready:
        print("Server not ready")
        sys.exit(1)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
        # Public health
        r = await get(client, "/health")
        assert_ok(r, 200)

        # OpenAPI available
        r = await get(client, "/openapi.json")
        assert_ok(r, 200)

        # Auth checks
        r = await client.get("/system/status")
        if r.status_code != 401:
            print("Expected 401 without API key")
            sys.exit(1)

        # System status
        r = await get(client, "/system/status", headers=HEADERS)
        assert_ok(r, 200)
        data = r.json()
        if "success" not in data:
            print("Invalid /system/status schema")
            sys.exit(1)

        # System config read empty ok
        r = await get(client, "/system/config", headers=HEADERS)
        assert_ok(r, 200)

        # System config write then read
        cfg_body = {
            "section": "app",
            "key": "test_key",
            "value": "test_value"
        }
        r = await put(client, "/system/config", json=cfg_body, headers=HEADERS)
        assert_ok(r, 200)
        r = await get(client, "/system/config", headers=HEADERS)
        assert_ok(r, 200)

        # Logs
        r = await get(client, "/system/logs?lines=5", headers=HEADERS)
        assert_ok(r, 200)

        # Stats
        r = await get(client, "/system/stats", headers=HEADERS)
        assert_ok(r, 200)

        # Backup create
        backup_body = {
            "backup_type": "full",
            "include_config": True,
            "include_data": False,
            "include_models": False
        }
        r = await post(client, "/system/backup", json=backup_body, headers=HEADERS)
        assert_ok(r, 200)

        # Restore non existing should 404
        restore_body = {
            "backup_file": "backups/missing_file.tar.gz",
            "restore_config": True,
            "restore_data": False,
            "restore_models": False
        }
        r = await post(client, "/system/restore", json=restore_body, headers=HEADERS)
        if r.status_code != 404:
            print("Expected 404 for missing backup file")
            print(r.status_code, r.text)
            sys.exit(1)

        # Clear cache
        r = await delete(client, "/system/cache", headers=HEADERS)
        assert_ok(r, 200)

        print("All smoke tests passed")


if __name__ == "__main__":
    asyncio.run(run_tests())
