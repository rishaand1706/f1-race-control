"""The explanation card rebuilt as a Semantic Kernel agent. See docs/concepts/09-agents-and-guardrails.md."""

# Must come before any semantic_kernel import: SK reads its telemetry switch at import time.
from agent import telemetry as _telemetry  # noqa: F401
