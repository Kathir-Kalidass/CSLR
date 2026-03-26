from __future__ import annotations

from typing import Any

from .constants import INIT_SEQUENCE, PIPELINE_ORDER
from .payloads import build_idle_payload, build_running_payload
from .stream_loop import run_stream_loop


class DemoEngine:
    def __init__(self) -> None:
        self.pipeline_order = PIPELINE_ORDER
        self.init_sequence = INIT_SEQUENCE

    def make_payload(self, tick: int, client_state: dict[str, Any]) -> dict[str, Any]:
        if not bool(client_state.get("running", False)):
            return build_idle_payload(
                client_state=client_state,
                init_sequence=self.init_sequence,
                pipeline_order=self.pipeline_order,
            )

        return build_running_payload(
            tick=tick,
            client_state=client_state,
            init_sequence=self.init_sequence,
            pipeline_order=self.pipeline_order,
        )

    async def stream(self, websocket: Any, client_state: dict[str, Any]) -> None:
        await run_stream_loop(websocket, client_state=client_state, make_payload=self.make_payload)


engine = DemoEngine()
