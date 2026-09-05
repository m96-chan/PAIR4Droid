# ADR-0003: Report accelerator utilisation as the EWMA of "inference in flight"

**Status:** accepted · 2026-09-05

## Context
PAIR ranks candidate nodes by `Pending` then `GPUPressure`; pressure is derived from the max
`GPUs[].utilization_percent` in node-info, banded at 40/70/85 % (see docs/pair-contract.md §2).
A node without telemetry gets the neutral band 1, so an idle phone ties with a 40–70 %-loaded
workstation. Android exposes no GPU/NPU utilisation counter comparable to NVML/DXGI.

## Decision
`pair-telemetry` emits one `GPUs[0]` entry named after the accelerator backend (e.g.
`"Adreno 750 (llama.cpp)"`) whose `utilization_percent` is an EWMA over ticks of
`active > 0 ? 100 : 0`, `vram_bytes` = the RAM budget for models, `vram_used_bytes` = bytes of
the loaded model. `telemetryValid` is true after two CPU samples.

## Consequences
+ Idle phone → 0 % → band 0 (better than neutral). Busy phone → climbs to band 3 → PAIR prefers other owners.
+ Also honest for `/api/ps` and the UI.
− It is a proxy, not a hardware counter; documented as such in the UI.
