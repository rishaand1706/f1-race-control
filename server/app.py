"""The browser-facing server.

    python -m server.app                 # then open http://127.0.0.1:8765
    python -m server.app --port 9000

Bound to 127.0.0.1 and nowhere else. The replay is FIA and F1 timing data run through a
local model; nothing here is meant to be reachable from another machine.

Four endpoints, because the browser only ever does three things - list races, start one,
and watch - plus a stop:

    GET  /api/races      cached 2026 races that can be replayed
    POST /api/replay     start (or restart) a replay; body: round, speed, from_lap, mode
    POST /api/stop
    GET  /api/stream     Server-Sent Events: a snapshot, then deltas
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import config
from engine.retrieve import Retriever
from ingest.replay import cached_races
from server.bus import Bus
from server.pipeline import Replay

STATIC = Path(__file__).parent / "static"
YEAR = 2026


class ReplayRequest(BaseModel):
    round: int
    speed: float = Field(20.0, ge=0.5, le=200)
    from_lap: int | None = Field(None, ge=1)
    mode: str = Field("pipeline", pattern="^(pipeline|inline)$")


class Hub:
    """The one current replay, and the bus every browser listens on."""

    def __init__(self) -> None:
        self.retriever: Retriever | None = None
        self.replay: Replay | None = None
        self.bus = Bus(snapshot=self.snapshot)
        self.races: list[dict] = []
        self.warming: asyncio.Task | None = None

    def snapshot(self) -> dict:
        if self.replay is None:
            return {"run": None, "state": None, "feed": [], "cards": []}
        return self.replay.snapshot()

    async def start(self, req: ReplayRequest) -> None:
        await self.stop()
        self.replay = Replay(self.retriever, self.bus.publish, year=YEAR, rnd=req.round,
                             speed=req.speed, from_lap=req.from_lap, mode=req.mode)
        task = self.replay.start()
        task.add_done_callback(self._report)

    async def stop(self) -> None:
        if self.replay is None:
            return
        # Always cancel: a replay whose events have all been emitted can still have a
        # prose worker mid-card, and it would carry on writing into the next race.
        await self.replay.stop()
        if self.replay.status in ("loading", "running"):
            self.replay.status = "stopped"
            self.bus.publish("run", self.replay.info())

    def _report(self, task: asyncio.Task) -> None:
        if task.cancelled() or task.exception() is None:
            return
        logging.exception("replay crashed", exc_info=task.exception())
        if self.replay is not None:
            self.replay.status = f"crashed: {type(task.exception()).__name__}"
            self.bus.publish("run", self.replay.info())


hub = Hub()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # ~30MB of vectors and BM25, loaded once. Per request it would cost more than a card.
    hub.retriever = await asyncio.to_thread(Retriever)
    hub.races = await asyncio.to_thread(cached_races, YEAR)
    # Load the card model now, not on the first incident: a generate call with no prompt
    # loads it and returns. Failure is not fatal - Ollama may be starting - so it is
    # fire-and-forget, and the first card pays the load if this did not land.
    hub.warming = asyncio.create_task(_warm_model())    # held: unreferenced tasks can be GC'd
    yield
    await hub.stop()


async def _warm_model() -> None:
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            await c.post(f"{config.OLLAMA_HOST}/api/generate",
                         json={"model": config.CARD_MODEL, "prompt": "",
                               "keep_alive": config.OLLAMA_KEEP_ALIVE})
    except httpx.HTTPError as exc:
        logging.warning("could not preload %s: %s", config.CARD_MODEL, exc)


app = FastAPI(title="F1 Race Control Copilot", lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/races")
async def races() -> list[dict]:
    return hub.races


@app.post("/api/replay", status_code=202)
async def replay(req: ReplayRequest) -> dict:
    if req.round not in {r["round"] for r in hub.races}:
        raise HTTPException(404, f"round {req.round} is not in the local cache")
    await hub.start(req)
    return {"started": req.model_dump()}


@app.post("/api/stop")
async def stop() -> dict:
    await hub.stop()
    return {"stopped": True}


@app.get("/api/stream")
async def stream() -> StreamingResponse:
    return StreamingResponse(
        hub.bus.subscribe(), media_type="text/event-stream",
        # no-transform and X-Accel-Buffering stop a proxy from buffering the stream
        # into one late lump, which would silently undo the point of streaming.
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    logging.getLogger("fastf1").setLevel(logging.WARNING)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
