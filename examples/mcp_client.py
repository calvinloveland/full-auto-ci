"""Example script demonstrating how to talk to the Full Auto CI MCP server."""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Dict


async def _send_json(writer: asyncio.StreamWriter, payload: Dict[str, Any]) -> None:
    writer.write(json.dumps(payload).encode("utf-8") + b"\n")
    await writer.drain()


async def main(host: str, port: int, token: str | None) -> None:
    reader, writer = await asyncio.open_connection(host, port)
    try:
        request = {"jsonrpc": "2.0", "id": 1, "method": "handshake", "params": {}}
        if token:
            request["params"]["token"] = token
        await _send_json(writer, request)
        response = await reader.readline()
        print("Handshake response:", json.loads(response.decode("utf-8")))

        list_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "listRepositories",
            "params": {"token": token} if token else {},
        }
        await _send_json(writer, list_request)
        response = await reader.readline()
        print("Repositories:", json.loads(response.decode("utf-8")))
    finally:
        writer.close()
        await writer.wait_closed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Talk to a Full Auto CI MCP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token")
    args = parser.parse_args()
    asyncio.run(main(args.host, args.port, args.token))
