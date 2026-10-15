"""One replay, many browsers: fan-out over Server-Sent Events.

Why SSE and not WebSockets. Everything here flows one way, server to browser; the browser
never says anything back except "start a replay", which is an ordinary POST. SSE is exactly
that shape: a long-lived HTTP response the server keeps writing to, with reconnection built
into the browser's `EventSource`. A WebSocket buys a bidirectional channel this app would
not use, and costs a protocol upgrade, a separate framing layer, and reconnection logic
written by hand.

Why snapshot-plus-deltas and not SSE's own resume. `EventSource` sends `Last-Event-ID` when
it reconnects, and a server can replay everything after that id. That needs every delta
kept, and most deltas here are single tokens of prose: thousands per race, worthless a
second later. The state they build is small - twenty cars and a few dozen cards - so any
client that connects or reconnects is sent a `snapshot` of the whole view and then deltas
from that point. Nothing to retain, and no gap: the snapshot is taken and the client is
registered in one synchronous step, so no event can fall between them.

Why a slow client is dropped rather than waited for. `publish` never awaits. If a browser
cannot keep up - a backgrounded tab, a bad network - its queue fills, and the choices are
to block the replay (every other viewer freezes), to buffer without limit (memory grows
until the process dies), or to cut that one client off. Cutting it off is the only one
that hurts nobody else, and it costs the client almost nothing: `EventSource` reconnects
and the snapshot brings it current.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

HEARTBEAT_S = 15.0     # proxies and some browsers close a silent stream after ~30-60s


@dataclass(eq=False)     # identity, not value: clients live in a set
class Client:
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=2000))
    dropped: bool = False


def frame(event: str, data: Any) -> str:
    """One SSE frame. `data` is one line of JSON, so it can never contain the blank line
    that would end the frame early."""
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


class Bus:
    def __init__(self, snapshot: Callable[[], dict]) -> None:
        self.snapshot = snapshot
        self.clients: set[Client] = set()
        self.dropped_total = 0

    def publish(self, event: str, data: Any) -> None:
        msg = frame(event, data)
        for c in list(self.clients):
            try:
                c.queue.put_nowait(msg)
            except asyncio.QueueFull:
                c.dropped = True
                self.clients.discard(c)
                self.dropped_total += 1

    async def subscribe(self) -> AsyncIterator[str]:
        client = Client()
        # Snapshot and registration with no await between them: nothing can be published
        # in the gap, so the client sees every delta after the state it was given.
        first = frame("snapshot", self.snapshot())
        self.clients.add(client)
        try:
            yield "retry: 2000\n\n" + first
            while not client.dropped:
                try:
                    yield await asyncio.wait_for(client.queue.get(), HEARTBEAT_S)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            self.clients.discard(client)
