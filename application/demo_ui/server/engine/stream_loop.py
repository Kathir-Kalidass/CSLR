from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from .frame_ingest import maybe_process_latest_client_frame


async def run_stream_loop(
    websocket: Any,
    client_state: dict[str, Any],
    make_payload: Callable[[int, dict[str, Any]], dict[str, Any]],
) -> None:
    tick = 0
    while True:
        maybe_process_latest_client_frame(client_state)
        client_state["tick"] = tick
        payload = make_payload(tick=tick, client_state=client_state)
        await websocket.send_text(json.dumps(payload))
        tick += 1
        await asyncio.sleep(0.35)
