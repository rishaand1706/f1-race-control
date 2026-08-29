"""The normalised event, and the one clock everything is placed on.

Replay (Stage 2) and live (Stage 6) read completely different sources: one reads a
finished session out of FastF1, the other tails a raw SignalR recording as it is written.
Everything downstream - the state reducer, the query builder, the explanation cards -
must not care which. So both are required to emit *this* type and nothing else.

That is the whole point of a normalised event schema: it is the seam that lets a fragile,
unavailable-most-of-the-week data source be swapped for a reliable one during development
without a single change to the code being developed.

Every event carries `t`, seconds since the session's own start. Not a wall-clock time, not
a pandas Timestamp: one float on one monotonic timeline. See
docs/concepts/05-event-streams.md for why the sources do not agree on that timeline and
how it is reconciled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Event kinds. Deliberately few: everything the app reacts to is one of these.
RACE_CONTROL = "race_control"      # the messages the app exists to explain
LAP_COMPLETED = "lap_completed"    # one car crossed the line
TRACK_STATUS = "track_status"      # green / yellow / SC / VSC / red
SESSION_STATUS = "session_status"  # started / aborted / finished


@dataclass(frozen=True)
class Event:
    t: float                          # seconds since session start
    kind: str
    lap: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.payload.get(key)

    @property
    def clock(self) -> str:
        """t rendered as h:mm:ss, for terminal output.

        The sign is taken out before the arithmetic. Python floors on negative division,
        so -3518s came out as "-1:01:22" instead of "-0:58:38" - and pre-race messages,
        which are all at negative t, are exactly the ones this renders.
        """
        sign = "-" if self.t < 0 else ""
        s = int(abs(self.t))
        return f"{sign}{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"

    def describe(self) -> str:
        if self.kind == RACE_CONTROL:
            flag = f"[{self['flag']}] " if self["flag"] else ""
            return f"{flag}{self['message']}"
        if self.kind == LAP_COMPLETED:
            lt = self["lap_time"]
            return (f"L{self.lap:<2} P{self['position'] or '?':<2} "
                    f"#{self['car']:<3} {self['driver']:<4} "
                    f"{_fmt_laptime(lt):>9}  {self['compound'] or '':<7}"
                    f"{'  PIT' if self['pit_in'] else ''}")
        if self.kind == TRACK_STATUS:
            return f"track: {self['label']}"
        if self.kind == SESSION_STATUS:
            return f"session: {self['status']}"
        return str(self.payload)


def _fmt_laptime(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{int(seconds // 60)}:{seconds % 60:06.3f}"


# FastF1 exposes track status as a numeric code. Naming them here, once, keeps the
# magic numbers out of the reducer and out of the UI.
TRACK_STATUS_LABELS = {
    "1": "AllClear",
    "2": "Yellow",
    "3": "Unknown",
    "4": "SafetyCar",
    "5": "Red",
    "6": "VSCDeployed",
    "7": "VSCEnding",
}
