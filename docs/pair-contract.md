# PAIR wire contract for a manual node

Source of truth for implementing a PAIR-compatible node (Rust / Android). Every claim cites
`path:line` in the NVIDIA Personal AI Router checkout at `/home/user/nvidia/personal-ai-router`
(commit `13b6811`, "Update to 0.1.1"). Paths below are relative to that root.

Terminology:

- **Manual node** — a host the user typed into PAIR. It is never discovered over mDNS; PAIR
  probes it directly (`services/nvpair-ui-broker/manualnodes.go:158-161`).
- **Peer node** — a host that advertises `_nvpair-node._tcp` and (for inference) is cluster-paired.
  See §4.
- **Broker** — `nvpair-ui-broker`, the supervisor that owns every worker subprocess and speaks
  newline-delimited JSON-RPC 2.0 to them over stdio.

The fastest path to being usable by PAIR is the **manual node** path (§1–§3): it needs only three
plain-HTTP servers and no mDNS, no TLS, and no cluster membership.

---

## 1. Manual node registration & probing

### 1.1 Two different `node/add*` methods

| Method | Speaker → listener | Purpose |
| --- | --- | --- |
| `node/add` | UI/desktop → broker → `nvpair-manual-nodes` | Register a user-added host to be probed |
| `node/add-manual` | broker → `ollama-proxy` / `lmstudio-proxy` | Install a routing target inside a proxy |

The broker relays `node/add`, `node/remove`, `nodes/list` verbatim to `nvpair-manual-nodes`
(`services/nvpair-ui-broker/broker.go:2897-2898`, `services/nvpair-ui-broker/broker.go:3146-3211`).
If no manual-nodes worker is supervised, the client gets JSON-RPC error
`-32000 "manual-nodes not available"` (`services/nvpair-ui-broker/broker.go:3153-3158`).

`node/add` uses the async relay (no timeout) because it kicks off an initial probe
(`services/nvpair-ui-broker/broker.go:3146-3151`).

### 1.2 `node/add` params — `ManualEntry`

```go
type ManualEntry struct {
    Address string `json:"address"`
    Name    string `json:"name"`
    TLSPort int    `json:"tls_port,omitempty"`
    MTLS    bool   `json:"mtls,omitempty"`
}
```
`services/nvpair-manual-nodes/manager.go:91-96`

- `address` is **required**; empty → `-32602 "address is required"`
  (`services/nvpair-manual-nodes/manager.go:651-654`).
- A params value that is not a JSON object → `-32602 "invalid params: expected {\"address\": \"...\"}"`
  (`services/nvpair-manual-nodes/manager.go:647-650`).
- `address` may be an IP literal or a hostname. Hostnames are re-resolved on every probe because
  the probe transport disables keep-alives (`services/nvpair-manual-nodes/manager.go:196-204`).
- `tls_port > 0` switches the node-info probe to `https://` on that port; `tls_port == 0` means
  plain HTTP on 14318 (`services/nvpair-manual-nodes/manager.go:257-280`).
- `mtls` is **informational only** on the entry — it changes nothing about how PAIR probes; it is
  persisted so the UI can render what the operator configured
  (`services/nvpair-manual-nodes/manager.go:86-90`).

There is **no** `id`, `host`, `port`, `addresses` or `models` field on `node/add`. Those belong to
`node/add-manual` (§1.7).

### 1.3 `nodeID(entry)`

```go
func nodeID(entry ManualEntry) string {
    if entry.Name != "" { return entry.Name }
    return "manual:" + entry.Address
}
```
`services/nvpair-manual-nodes/manager.go:697-702`

Consequences for an implementer:

- The manual id is the user's `name`, or the literal string `"manual:" + address`.
- Re-adding the same `{address, name}` overwrites the tracked entry in place
  (`services/nvpair-manual-nodes/manager.go:548-550`) — no duplicate.
- Re-adding the same address under a **different** `name` produces a **second, independent** manual
  entry with a different id. Both probe the same host; the broker collapses them downstream by
  `hostUuid` (§2.5).
- The manual id is only the *initial* operational key. As soon as `/v1/node-info` reports a
  `hostUuid`, the broker rekeys the node to that UUID
  (`services/nvpair-ui-broker/manualnodes.go:71-97`, `services/nvpair-ui-broker/broker.go:1265-1288`).

### 1.4 Probe loop timing

```go
probeInterval      = 10 * time.Second
probeTimeout       = 3  * time.Second
probeFailThreshold = 3
```
`services/nvpair-manual-nodes/manager.go:28-44`

- `probeTimeout` is the whole-request `http.Client.Timeout` for every probe
  (`services/nvpair-manual-nodes/manager.go:184-192`).
- The ticker fires every `probeInterval` and probes each tracked entry **sequentially** on one
  goroutine (`services/nvpair-manual-nodes/manager.go:220-248`).
- `node/add` responds immediately with a stub status and probes once asynchronously; when that first
  probe finishes it emits `node/discovered` with the real status
  (`services/nvpair-manual-nodes/manager.go:531-566`).

### 1.5 The three (four) probe requests

Executed per node per tick, in this order
(`services/nvpair-manual-nodes/manager.go:250-281`):

| # | Request | Success criterion | Yields |
| --- | --- | --- | --- |
| 1 | `GET http://<addr>:11434/` | HTTP **200** exactly | `ollama_up` |
| 2 | `GET http://<addr>:11434/api/tags` | 200 + parseable JSON | `ollama_models[]` from `models[].name` |
| 3 | `GET http://<addr>:1234/v1/models` | HTTP **200** exactly | `lmstudio_up`, `lmstudio_models[]` from `data[].id` |
| 4 | `GET http(s)://<addr>:14318\|<tls_port>/v1/node-info` | 200 + JSON decodes into `NodeInfoResponse` | GPUs/CPU/memory/telemetry/hostUuid |

Details:

- **Ollama liveness** is `GET /` on 11434; any non-200 (or transport error) marks the node down and
  `/api/tags` is not even attempted (`services/nvpair-manual-nodes/manager.go:449-471`). The body of
  `GET /` is ignored — it is closed without reading
  (`services/nvpair-manual-nodes/manager.go:458`).
- **`/api/tags`** failure (transport error, non-200, or JSON decode error) yields `nil` models but
  does **not** flip `ollama_up` back to false
  (`services/nvpair-manual-nodes/manager.go:473-497`). Model names come from
  `models[].name` only — `models[].model` is not read here.
- **LM Studio**: one `GET /v1/models` doubles as liveness and inventory. A 200 whose body fails to
  decode still reports **up** with no models
  (`services/nvpair-manual-nodes/manager.go:431-436`). Only non-empty `data[].id` values are kept
  (`services/nvpair-manual-nodes/manager.go:437-442`).
- **node-info**: transport error, non-200, **or** a JSON decode failure all yield
  `node_info_up = false` and a zero-valued `NodeInfoResponse`
  (`services/nvpair-manual-nodes/manager.go:499-529`).

### 1.6 Compile-time constant ports

| Port | Constant | Citation |
| --- | --- | --- |
| 11434 (Ollama) | hardcoded literal at both call site and status | `services/nvpair-manual-nodes/manager.go:254`, `:288`, `:450`, `:474`, `:542` |
| 1234 (LM Studio) | `const lmStudioPort = 1234` | `services/nvpair-manual-nodes/manager.go:400-404` |
| 14318 (node-info, plaintext) | hardcoded literal | `services/nvpair-manual-nodes/manager.go:264`, `:534` |
| `tls_port` (node-info, HTTPS) | per-entry, defaults 14319 on the server side | `services/nvpair-manual-nodes/manager.go:266-269`, `services/nvpair-node-info/main.go:298` |

**None of the first three are configurable per node.** An Android node MUST bind 11434 (Ollama
dialect) and/or 1234 (OpenAI dialect), plus 14318 for node-info, or it will never be probed
successfully.

### 1.7 Probe result — `ManualNodeStatus`

```go
type ManualNodeStatus struct {
    ID             string      `json:"id"`
    Name           string      `json:"name,omitempty"`
    Address        string      `json:"address"`
    OllamaUp       bool        `json:"ollama_up"`
    OllamaPort     int         `json:"ollama_port"`      // always 11434
    OllamaModels   []string    `json:"ollama_models,omitempty"`
    LMStudioUp     bool        `json:"lmstudio_up"`
    LMStudioPort   int         `json:"lmstudio_port"`    // always 1234
    LMStudioModels []string    `json:"lmstudio_models,omitempty"`
    NodeInfoUp     bool        `json:"node_info_up"`
    NodeInfoPort   int         `json:"node_info_port"`
    TLSEnabled     bool        `json:"tls_enabled,omitempty"`
    MTLSRequired   bool        `json:"mtls_required,omitempty"`
    GPUs           []GPUInfo   `json:"gpus,omitempty"`
    CPU            *CPUInfo    `json:"cpu,omitempty"`
    Memory         *MemoryInfo `json:"memory,omitempty"`
    TelemetryValid bool        `json:"telemetryValid"`
    MSSince        int64       `json:"msSince"`
    HostUUID       string      `json:"hostUuid,omitempty"`
}
```
`services/nvpair-manual-nodes/manager.go:105-131`

Emitted as the result of `node/add`, as elements of `nodes/list` `{"nodes":[…]}`, and as the params
of the `node/discovered` / `node/updated` / `node/removed` notifications
(`services/nvpair-manual-nodes/manager.go:349`, `:562`, `:577`, `:655-658`, `:677-681`).

`node/updated` fires only when a watched field changes; the change set includes
`ollama_up`, `lmstudio_up`, `node_info_up`, `hostUuid`, both model lists, GPUs, CPU, memory,
`telemetryValid` and `msSince` (`services/nvpair-manual-nodes/manager.go:332-354`). Because
`msSince` is included and node-info reports the age of its last GPU sample in ms, a node that
reports live telemetry will produce a `node/updated` on essentially every probe.

### 1.8 Failure, eviction, and re-add

**A manual node is never evicted by the prober.** `probeFailThreshold` only surfaces an error:

```
// This surfaces an error; it does not evict, so it is deliberately shorter
// than the ~60 s shared/discovery waits before dropping an mDNS node.
```
`services/nvpair-manual-nodes/manager.go:32-43`

- `reachable = ollama_up || lmstudio_up || node_info_up`
  (`services/nvpair-manual-nodes/manager.go:305`).
- Any reachable probe resets `consecutiveFails` to 0; otherwise it increments
  (`services/nvpair-manual-nodes/manager.go:324-329`).
- On the exact probe where the counter first reaches 3 (≈30 s), one `errors:report` notification is
  emitted with id `manual-nodes:probe-failed:<node-id>`, severity `warning`, action `none`
  (`services/nvpair-manual-nodes/manager.go:366-382`, `services/nvpair-manual-nodes/manager.go:392-398`).
  If the address is an IP literal, the message appends advice about DHCP reassignment
  (`services/nvpair-manual-nodes/manager.go:372-374`).
- Recovery (a reachable probe while above threshold) emits `errors:clear` for the same id
  (`services/nvpair-manual-nodes/manager.go:383-388`).
- The last-learned `hostUuid` is preserved across a node-info blip so the node is not rekeyed
  (`services/nvpair-manual-nodes/manager.go:316-322`).
- `node/remove {"id":"…"}` → `{"removed":bool}`, emits `node/removed` plus a defensive
  `errors:clear` (`services/nvpair-manual-nodes/manager.go:568-590`, `:661-675`).
- **State is in-memory only.** A `nvpair-manual-nodes` restart loses every entry; the client must
  re-add (`services/nvpair-ui-broker/README.md:478`). The desktop persists them itself and replays
  `node/add` (`desktop/src/electron/service-bridge/modular-supervisor.ts:845-856`).

While a node is down, the broker pulls it out of both proxies with `node/remove-manual` so no
inference is routed to it (`services/nvpair-ui-broker/manualnodes.go:186-191`).

### 1.9 The manual → proxy bridge (`node/add-manual`)

When the broker supervises both `nvpair-manual-nodes` and a proxy, each probe result is bridged
per-engine (`services/nvpair-ui-broker/manualnodes.go:162-191`):

- `ollama_up == true` → `ollama-proxy` `node/add-manual` with `port = ollama_port`
- `lmstudio_up == true` → `lmstudio-proxy` `node/add-manual` with `port = lmstudio_port`
- otherwise → that proxy's `node/remove-manual`

Payload (`services/nvpair-ui-broker/manualnodes.go:141-148`, `:177-186`):

```json
{"id":"<hostUuid or manual id>","host":"<address>","port":11434,"addresses":["<address>"],"models":["…"]}
```

The proxy validates: `id` non-empty, `port != 0`, `len(addresses) >= 1`, else
`-32602 "id, port, and at least one address are required"`
(`services/ollama-proxy/proxy.go:2231-2240`, `services/lmstudio-proxy/proxy.go:1974-1983`).
Result is `{"added":bool}` — `false` means an existing entry was updated
(`services/ollama-proxy/discovery.go:145-151`). The proxy then emits `node/discovered` or
`node/updated` with the node stamped with a canonical `ip`
(`services/ollama-proxy/proxy.go:2245-2251`, `services/ollama-proxy/discovery.go:61-66`).

The proxy's `Node` wire shape (`services/ollama-proxy/discovery.go:33-59`):

```go
type Node struct {
    ID        string   `json:"id"`
    Host      string   `json:"host"`
    Port      int      `json:"port"`
    Addresses []string `json:"addresses"`
    TXT       []string `json:"txt"`
    Models    []string `json:"models,omitempty"`
    IP        string   `json:"ip,omitempty"`
    ClusterUUID string `json:"-"`   // internal, never on the wire
}
```

Manual nodes live in a separate overlay from relay-discovered nodes and are merged at read time,
relay entries winning on id collision (`services/ollama-proxy/discovery.go:86-103`).
`node/remove-manual {"id":…}` → `{"removed":bool}`, and clears the user selection if it pointed at
that node (`services/ollama-proxy/proxy.go:2253-2274`).

---

## 2. `GET /v1/node-info`

### 2.1 What `nvpair-node-info` actually emits

```go
type GPUInfo struct {
    Name               string `json:"name"`
    VramBytes          uint64 `json:"vram_bytes,omitempty"`
    VramUsedBytes      uint64 `json:"vram_used_bytes,omitempty"`
    UtilizationPercent uint32 `json:"utilization_percent,omitempty"`
}
type CPUInfo struct {
    Name               string `json:"name,omitempty"`
    Cores              uint32 `json:"cores,omitempty"`
    UtilizationPercent uint32 `json:"utilization_percent,omitempty"`
}
type MemoryInfo struct {
    TotalBytes uint64 `json:"total_bytes,omitempty"`
    UsedBytes  uint64 `json:"used_bytes,omitempty"`
}
type NodeInfoResponse struct {
    GPUs           []GPUInfo   `json:"GPUs"`
    CPU            *CPUInfo    `json:"cpu,omitempty"`
    Memory         *MemoryInfo `json:"memory,omitempty"`
    TelemetryValid bool        `json:"telemetryValid"`
    MSSince        int64       `json:"msSince"`
    HostUUID       string      `json:"hostUuid,omitempty"`
    ClusterUUID    *string     `json:"clusterUuid,omitempty"`
}
```
`services/nvpair-node-info/main.go:30-49`, `:59-63`, `:69-72`, `:74-102`

Field-by-field:

| JSON name | Type | Unit | Optional | Notes |
| --- | --- | --- | --- | --- |
| `GPUs` | array | — | **required key** (no `omitempty`); may be `[]` or `null` | Capital "GPUs" — the only PascalCase key in the payload |
| `GPUs[].name` | string | — | required key | Free text; the desktop sorts NVIDIA-named GPUs first (`desktop/src/electron/service-bridge/modular-state.ts:496-513`) |
| `GPUs[].vram_bytes` | uint64 | bytes | omit when 0 | Total VRAM; on unified-memory hosts, total system RAM (`services/nvpair-node-info/README.md:86`) |
| `GPUs[].vram_used_bytes` | uint64 | bytes | omit when 0 | |
| `GPUs[].utilization_percent` | uint32 | percent 0–100 | omit when 0 | **This is the only field that feeds scheduling.** |
| `cpu` | object | — | whole object omitted when unknown | Pointer + `omitempty` |
| `cpu.name` | string | — | omit when empty | |
| `cpu.cores` | uint32 | count | omit when 0 | |
| `cpu.utilization_percent` | uint32 | percent | omit when 0 | Display-only |
| `memory` | object | — | whole object omitted when total is 0 | |
| `memory.total_bytes` | uint64 | bytes | omit when 0 | |
| `memory.used_bytes` | uint64 | bytes | omit when 0 | |
| `telemetryValid` | bool | — | **always present** | true once a usable GPU utilization sample exists |
| `msSince` | int64 | milliseconds | **always present** | Age of that sample at response time; 0 when invalid |
| `hostUuid` | string | — | omit when empty | Stable per-host identity |
| `clusterUuid` | string \| absent | — | pointer + omitempty — **three states** | absent = unknown; `""` = unclustered; value = principal |

Assembly and semantics:

- `buildResponseAt` merges the static inventory with the latest collector snapshot
  (`services/nvpair-node-info/main.go:176-212`). `cpu` is emitted only when static CPU detection
  succeeded (`:199-203`); `memory` only when `memTotal > 0` (`:204-209`).
- `telemetryStatus` (`services/nvpair-node-info/main.go:214-223`): a zero sample timestamp →
  `(false, 0)`; otherwise `(true, age.Milliseconds())` with negative ages clamped to 0.
- Collector tick is 1 s on every supported OS (`services/nvpair-node-info/stats_linux.go:44`,
  `services/nvpair-node-info/stats_darwin.go:20`, `services/nvpair-node-info/stats_windows.go:94-100`).
  Platforms with no collector publish a zero snapshot → `telemetryValid:false`
  (`services/nvpair-node-info/stats_other.go:24`, `services/nvpair-node-info/stats.go:31-35`).
- Handler sets `Content-Type: application/json` and, when the process runs with `--cluster-dir`
  **and** is a cluster member, requires a pinned client cert else 403
  (`services/nvpair-node-info/main.go:260-272`).
- Listener default is `:14318` plain, `:14319` for BYO-TLS
  (`services/nvpair-node-info/main.go:297-302`), `ReadHeaderTimeout` 5 s
  (`services/nvpair-node-info/main.go:440-444`).
- The canonical example body is in `services/nvpair-node-info/README.md:52-75`.

### 2.2 The subset `nvpair-manual-nodes` parses

```go
type NodeInfoResponse struct {
    GPUs           []GPUInfo   `json:"GPUs"`
    CPU            *CPUInfo    `json:"cpu,omitempty"`
    Memory         *MemoryInfo `json:"memory,omitempty"`
    TelemetryValid bool        `json:"telemetryValid"`
    MSSince        int64       `json:"msSince"`
    HostUUID       string      `json:"hostUuid,omitempty"`
}
```
`services/nvpair-manual-nodes/manager.go:69-81`

**`clusterUuid` is not read on the manual path.** The scanner (peer path) does read it, as a
pointer, to distinguish absent from empty (`services/nvpair-node-scanner/scanner.go:32-53`).

Unknown extra fields are ignored (`encoding/json` default) — safe to send more. Note that the
desktop also reads a non-Go field `inference_hardware_ids` when polling node-info directly
(`desktop/src/electron/service-bridge/modular-state.ts:1230`); the Go service does not emit it,
so it is optional.

### 2.3 UI consumption (brief)

The desktop polls `/v1/node-info` itself, per node, at `nodeInfoPort`
(`desktop/src/electron/service-bridge/modular-state.ts:1203-1212`). `mergeNodeInfoResponse`
(`:1214-1244`):

1. If the response's `hostUuid` is non-empty and differs from the node it polled, the merge is
   **skipped** — telemetry is never attributed to the wrong host (`:1221-1225`).
2. `GPUs` → `{name, vramBytes, vramUsedBytes, utilizationPercent}` (`:500-514`),
   `cpu` → `{name, cores, utilizationPercent}` (`:516-524`),
   `memory` → `{totalBytes, usedBytes}` (`:526-533`).
3. `nodeInfoUp` is set true on any successful merge (`:1242`).

Note the broker's own `AvailableNode` (the `discovery:get-nodes` / `discovery:nodes-changed` wire
shape) deliberately does **not** carry GPU/CPU/memory
(`services/nvpair-ui-broker/broker.go:70-115`, `services/nvpair-ui-broker/discovery.go:301-327`);
hardware detail reaches the UI through the direct node-info poll and through
`nodes/list` on `nvpair-manual-nodes`.

### 2.4 How `utilization_percent` becomes `GPUPressure`

The chain, end to end:

**Step 1 — node → one scalar.** Both ingest paths reduce the GPU array to `max(utilization_percent)`:

- Manual path: `services/nvpair-ui-broker/manualnodes.go:49-62`
  ```go
  for i := range status.GPUs {
      if status.GPUs[i].UtilizationPercent > utilization { utilization = … }
  }
  ```
- Peer path (scanner): `services/nvpair-node-scanner/daemon.go:1637-1642`, with the sample age
  advanced by the request's own elapsed time (`:1624-1636`).

The emitted value is `noderec.NodeTelemetry`
(`services/shared/noderec/noderec.go:616-624`):

```json
{"hostUuid":"…","gpuUtilizationPercent":42,"telemetryValid":true,"msSince":137}
```

**Step 2 — broker cache with source precedence.** `telemetryCache.Upsert` records per-source
observations; the projection prefers the **scanner** and falls back to **manual**
(`services/nvpair-ui-broker/telemetry.go:23-57`, `:71-90`). `observedTelemetryAt` ages `msSince`
forward by the time since receipt, and zeroes it when `telemetryValid` is false
(`services/nvpair-ui-broker/telemetry.go:132-151`). The projection is forwarded to the scheduler as
`scheduler:telemetry` (`services/shared/schedulerwire/schedulerwire.go:9`,
`services/nvpair-ui-broker/telemetry.go:177-188`), and replayed on scheduler restart (`:193-206`).

**Step 3 — scheduler EWMA + bands.** `services/nvpair-job-scheduler/telemetry.go:14-18`:

```go
gpuTelemetryFreshness = 10 * time.Second
gpuEWMAAlpha          = 0.35
unknownGPUPressure    = 1
```

`applyTelemetryAt` (`services/nvpair-job-scheduler/telemetry.go:41-74`):

- Drops the sample entirely when `hostUuid` is empty (`:42-44`).
- `incomingFresh = telemetryValid && normalizedAge <= 10s` (`:45-46`). `normalizedTelemetryAge`
  clamps a negative/zero to 0 and anything over 10 s to `10s + 1ms`, i.e. permanently stale
  (`:101-109`).
- Utilization is clamped to 100 (`:47-50`).
- If not fresh, the EWMA is **not** updated and `valid` is set false — the node's effective pressure
  falls back to `unknownGPUPressure = 1` (`:58-71`, `:83-88`).
- First fresh sample (or first after a staleness gap) seeds `ewma = utilization` and takes the raw
  band; subsequent fresh samples use `ewma = 0.35·u + 0.65·ewma_prev` with hysteresis (`:61-70`).

Bands (`services/nvpair-job-scheduler/telemetry.go:111-122`):

| EWMA utilization | GPUPressure |
| --- | --- |
| `< 40` | 0 |
| `40 … < 70` | 1 |
| `70 … < 85` | 2 |
| `>= 85` | 3 |

Hysteresis (`services/nvpair-job-scheduler/telemetry.go:124-138`): step up at `{40, 70, 85}`,
step down at `{—, 35, 65, 80}`.

Freshness decays without new samples: `telemetryFreshAt` adds wall-clock elapsed since receipt to
the age-at-receipt and compares against 10 s
(`services/nvpair-job-scheduler/telemetry.go:90-99`). So a node that stops answering silently
drifts to pressure 1 after ≈10 s.

**Step 4 — ranking.** `rankAt` (`services/nvpair-job-scheduler/schedule.go:82-127`) builds one
`NodeRank{ID, Pending, GPUPressure, Rank}` per node in the current node universe and sorts by:

1. `Pending + GPUPressure` ascending,
2. then `GPUPressure` ascending,
3. then `ID` ascending (stable tiebreak).

The result is emitted as `schedule:priority` per engine (`ollama`, `lmstudio`) only when the ranking
changed (`services/nvpair-job-scheduler/schedule.go:15-17`, `services/nvpair-job-scheduler/schedule.go:134-152`).

### 2.5 Does a manual node's node-info feed pressure? **Yes.**

Manual telemetry is a first-class source, not a no-op:

- `upsertManualNode` calls `b.ingestTelemetryAt(sourceManual, manualNodeTelemetry(s, key), receivedAt)`
  on every manual-node event (`services/nvpair-ui-broker/broker.go:1265-1277`), and again when a
  surviving alias is re-projected (`services/nvpair-ui-broker/broker.go:1313-1329`).
- The cache projection returns the manual observation **whenever no scanner observation exists for
  that `hostUuid`** (`services/nvpair-ui-broker/telemetry.go:48-57`). A host discovered over mDNS
  *and* added manually uses the scanner's value.
- Removing the last manual alias emits an invalid observation so the scheduler clears the node at
  once instead of waiting for freshness expiry
  (`services/nvpair-ui-broker/telemetry.go:93-117`, `services/nvpair-ui-broker/broker.go:1326-1328`).
- The manual node must also be **in the scheduler's node universe**, which is keyed by `hostUuid`
  from `discovery:nodes-changed` (`services/nvpair-job-scheduler/state.go:33-47`, `:65-88`).
  `manualToEnriched` guarantees a non-empty key: the reported `hostUuid` if present, else the manual
  id (`services/nvpair-ui-broker/manualnodes.go:71-97`), and the store projects it onto
  `AvailableNode.hostUuid` (`services/nvpair-ui-broker/discovery.go:313-326`).
- Telemetry for a `hostUuid` no longer in the node set is dropped on the next snapshot
  (`services/nvpair-job-scheduler/state.go:80-84`).

### 2.6 What `telemetryValid` gates — summary

| Consumer | Behavior when `telemetryValid == false` |
| --- | --- |
| `nvpair-node-info` itself | reports `msSince: 0` (`services/nvpair-node-info/main.go:214-217`) |
| Scanner (peer path) | `age = 0`, telemetry still emitted (`services/nvpair-node-scanner/daemon.go:1624-1648`) |
| Broker cache | `msSince` forced to 0, not aged forward (`services/nvpair-ui-broker/telemetry.go:133-137`) |
| Scheduler | sample is not folded into the EWMA; node's effective pressure = `unknownGPUPressure` **1** (`services/nvpair-job-scheduler/telemetry.go:46`, `:58-71`, `:83-88`) |

### 2.7 Minimum a manual node must return

**(a) To be shown in the UI with CPU/memory/GPU info:**

- `GET http://<addr>:14318/v1/node-info` → 200, `Content-Type: application/json`, body decoding into
  the struct in §2.2. `GPUs` may be `[]`.
- Include `cpu` and `memory` objects to populate those cards; omit them and the node card shows no
  CPU/RAM (that is exactly why they are pointer-optional —
  `services/nvpair-manual-nodes/manager.go:53-57`).
- Include `hostUuid` so the node is keyed by a stable identity and deduped against an mDNS record of
  the same machine (`services/nvpair-manual-nodes/manager.go:75-80`,
  `services/nvpair-ui-broker/manualnodes.go:74-81`). Also required for the desktop's direct poll to
  accept the merge (`desktop/src/electron/service-bridge/modular-state.ts:1221-1225`).
- Separately, `GET http://<addr>:11434/` must be 200 and/or `GET http://<addr>:1234/v1/models` must
  be 200 for the node to be "reachable" at all
  (`services/nvpair-manual-nodes/manager.go:305`).

**(b) To influence scheduling:**

- `hostUuid` non-empty (empty → the scheduler drops the telemetry outright,
  `services/nvpair-job-scheduler/telemetry.go:42-44`).
- `telemetryValid: true`.
- `msSince` ≤ 10000 at the moment the broker forwards it — the broker adds its own elapsed time on
  top (`services/nvpair-ui-broker/telemetry.go:141-149`), and the probe cadence is 10 s, so keep
  `msSince` well under a second (node-info's own collector ticks at 1 s).
- At least one entry in `GPUs` with a meaningful `utilization_percent`; the maximum across the array
  is what is used. A node reporting `GPUs: []` yields utilization 0 → band 0 (lowest pressure,
  *most* attractive to the scheduler) as long as `telemetryValid` is true. Reporting
  `telemetryValid: false` instead yields the neutral pressure 1.

Minimal scheduling-visible body:

```json
{"GPUs":[{"name":"Adreno 750","utilization_percent":37}],
 "telemetryValid":true,"msSince":80,
 "hostUuid":"8661676a-0d1c-4bd3-ac5e-4d370e6f1a9c"}
```

---

## 3. Inference forwarding to a manual node

Both proxies share one architecture. Each binds one TCP port and splits it by first byte
(`services/ollama-proxy/proxy.go:582-613`): plaintext HTTP → `handlePlain` (loopback-only, full
router), TLS → `handleClusterIngress` (pin-gated, terminal forward to the local engine).
Defaults: `ollama-proxy` on 11435 (`services/ollama-proxy/main.go:31`), `lmstudio-proxy` on 1234
(`services/lmstudio-proxy/main.go:22`).

**Neither proxy has an HTTP mux on the routing path.** `handlePlain` → `handleHTTP` handles *every*
path (`services/ollama-proxy/proxy.go:584-585`, `services/ollama-proxy/ingress.go:63-85`). Only the
model-list routes are special-cased; everything else is passed to `httputil.ReverseProxy`.

### 3.1 OpenAI lane — `lmstudio-proxy`

**Paths.**

- Special-cased: `GET /v1/models` → concurrent fan-out across all candidates, merged
  (`services/lmstudio-proxy/proxy.go:958-974`, `:816-932`).
- Tracked as workloads (POST only): `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`
  (`services/lmstudio-proxy/proxy.go:151-166`).
- **Every other path and method is forwarded unchanged** through the failover loop. There is no
  allowlist.

**Body handling.** The request body is read fully into memory once and replayed on each failover
attempt (`services/lmstudio-proxy/proxy.go:199-215`, `:1126-1128`). It is **not rewritten** — only
the top-level `"model"` string is *read* out for routing and workload attribution
(`services/lmstudio-proxy/proxy.go:208-214`). A body that fails to parse as JSON still forwards
verbatim with an empty model.

**Streaming / SSE.** The proxy uses `httputil.ReverseProxy` with only `Director`, `Transport`,
`ModifyResponse` and `ErrorHandler` set (`services/lmstudio-proxy/proxy.go:1132-1250`). Therefore:

- It **does not parse `data:` lines**, does not look for `[DONE]`, and does not require
  `Content-Type: text/event-stream`. Grepping the whole services tree finds no occurrence of
  `text/event-stream`, `[DONE]`, or `FlushInterval` outside an unrelated workload-store constant.
- Flushing is Go's stock `ReverseProxy` behavior: immediate per-write flush when the response is
  `text/event-stream` **or** when `Content-Length` is unset (chunked). Setting either is sufficient;
  a chunked NDJSON or SSE stream both flush per chunk.
- Each client write is bounded by a 30 s idle write deadline, reset after every successful write, so
  a slow generation is never penalized but a dead client is dropped
  (`services/lmstudio-proxy/proxy.go:252-281`, `:537` `idleClientWriteTimeout`).
  `FlushError` arms the same deadline around the flush
  (`services/lmstudio-proxy/proxy.go:283-315`).
- Every successful body write also reports node liveness upstream, but only once the response has
  committed (`services/lmstudio-proxy/proxy.go:1160-1167`, `:273-279`).

**Headers.** No explicit request-header manipulation exists (`grep 'req.Header'` in both
`proxy.go` files returns nothing). So stock `ReverseProxy` semantics apply: hop-by-hop headers
stripped, `X-Forwarded-For` appended. The `Director` rewrites scheme, `URL.Host`, and `req.Host` to
the target (`services/lmstudio-proxy/proxy.go:1133-1137`), so the upstream sees its own address in
`Host`. On the response side:

- An engine-declared `Access-Control-Allow-Origin` is preserved untouched; an engine that sets none
  gets the proxy's wildcard policy, and a stray `Access-Control-Allow-Credentials` with no origin is
  dropped (`services/lmstudio-proxy/proxy.go:1168-1176`, `services/shared/cors/cors.go:26-41`).
- An upstream `OPTIONS` response with no CORS policy is replaced by a local 204
  (`services/lmstudio-proxy/proxy.go:1151-1154`, `services/shared/cors/cors.go:67-80`).
- The model-list fan-out builds **fresh** requests (`Accept: application/json` only), so client
  credentials never reach a fan-out target — asserted by
  `services/lmstudio-proxy/failover_test.go:374-375`.

**Timeouts** (`services/lmstudio-proxy/proxy.go:512-523`, `:716-731`):

| Setting | Value |
| --- | --- |
| dial (`net.Dialer.Timeout`) | 10 s |
| TCP keep-alive | 30 s |
| `ResponseHeaderTimeout` | 120 s |
| idle conn timeout | 90 s / max 50 idle conns |
| inbound `ReadHeaderTimeout` | 10 s |
| inbound `IdleTimeout` | 90 s |
| client-write idle deadline | 30 s |
| model-list client (`Timeout` and `ResponseHeaderTimeout`) | 10 s; body capped at 16 MiB |

There is **no total request timeout** — a long generation is bounded only by the 120 s
response-header budget and the per-write idle deadline.

**Failover statuses.** `shouldRetry` (`services/lmstudio-proxy/proxy.go:1007-1019`,
`services/ollama-proxy/proxy.go:1202-1214` — identical):

```
408 Request Timeout      → retry
429 Too Many Requests    → retry
502 Bad Gateway          → retry
503 Service Unavailable  → retry
504 Gateway Timeout      → retry
404 Not Found            → retry ONLY on an inference request (POST to an inference path)
>= 500 (anything else)   → retry
everything else (400/401/403/422…) → returned to the client as-is
```

Plus: a transport/dial error always fails over if candidates remain, and forgets the node's
confirmed address so a multi-homed peer re-probes
(`services/lmstudio-proxy/proxy.go:1207-1225`).

Retry is only possible **before the first byte reaches the client**. `ModifyResponse` fires after
status+headers but before the body streams; returning the `retrySignal` sentinel aborts and
advances (`services/lmstudio-proxy/proxy.go:1141-1150`, `:1116-1122`). The last candidate never
retries (`last := i == len(candidates)-1`, `:1125`, `:1145`).

When every candidate fails at the transport, the client gets a single `502` with body
`{"error":"upstream error: …"}` plus CORS and `X-Content-Type-Options: nosniff`
(`services/lmstudio-proxy/proxy.go:1226-1249`). When no candidate exists at all, the client gets a
local `502` with `{"error":"no active node selected or available"}`, or
`{"error":"no available node advertises the requested model"}` for a model-bearing inference request
(`services/lmstudio-proxy/proxy.go:975-1001`).

The cross-process proof of the 404-owner → 200-owner failover is
`services/tests/model_routing_interop_test.go:41-151`: four manual nodes are installed, priority is
set so the 404 owner is tried first, and the test asserts the 404 owner and the 200 owner are each
hit exactly once while the ineligible nodes are never touched (`:112-132`).

**Owner selection (Capability Gate).** `resolveCandidates(model)`
(`services/lmstudio-proxy/proxy.go:1331-1461`):

1. Cluster membership and pins are re-derived per request (`:1346`).
2. If `model != ""`, the node set is filtered to advertised owners via `nodeAdvertisesModel`
   (`:1350-1358`). This runs **before** selection and priority — a user-pinned node that does not
   advertise the model is excluded (proven by
   `services/ollama-proxy/failover_test.go:528-586`).
3. The remaining nodes are sorted by ID for stability (`:1359-1363`).
4. Candidates are appended in three passes: explicit `node/select`, then scheduler priority order,
   then everything else by ID (`:1439-1455`).
5. Per node, the dial target is chosen: self → the local backend; pinned cluster peer → `https://`
   with `peerUUID` set; **manual node → plain `http://` to the user-supplied address**; unpinned
   relay peer → **dropped** (`:1386-1421`). Self-targets and duplicate resolved hosts are dropped
   (`:1422-1431`).
6. For a model-bearing inference request, `reserveCandidate` then moves the least-loaded
   scheduler-listed candidate to the front, load = `pending + gpuPressure + local reservations`
   (`:1472-1526`).

`services/lmstudio-proxy/ingress_test.go:16-37` is the load-bearing assertion: on an unclustered
node, **only** the manual node survives as a candidate, dialed plaintext with no `peerUUID`.

### 3.2 Ollama lane — `ollama-proxy`

Everything in §3.1 applies identically (same `statusCapture`, same `shouldRetry`, same
`resolveCandidates` shape, same timeout constants at
`services/ollama-proxy/proxy.go:528-539`, `:870-885`), with these differences:

**Special-cased paths.** `GET /api/tags` **and** `GET /v1/models` are both fanned out and merged
(`services/ollama-proxy/proxy.go:1153-1169`, `:981-1127`). The response envelope matches the request
route: `{"models":[…]}` for `/api/tags`, `{"object":"list","data":[…]}` for `/v1/models`
(`:1109-1120`). An empty merged inventory returns exactly `{"models":[]}`
(`services/ollama-proxy/failover_test.go:519-521`). If *no* candidate returned a valid list, the
response is `503` with `{"error":"model inventory unavailable"}` (`:1104-1108`).

**Everything else is forwarded verbatim** through the failover loop — including `/api/chat`,
`/api/generate`, `/api/show`, `/api/ps`, `/api/version`, `/api/embed`, `/api/embeddings`, `/api/pull`,
and `GET /`. There is no per-path table; the only routing decision is whether the request counts as
a workload (`services/ollama-proxy/proxy.go:152-166`):

```
POST /api/generate, /api/chat, /api/embeddings, /api/embed,
POST /v1/chat/completions, /v1/completions, /v1/embeddings
```

Anything else (including `GET /api/version`) is forwarded but is not model-gated and does **not**
get 404-failover — proven by `services/ollama-proxy/failover_test.go:358-363` (`GET /api/version`
returning 404 is passed straight through).

**`GET /`** is not handled specially anywhere in the proxy — it is forwarded like any other path. It
matters only to `nvpair-manual-nodes`, which requires a bare `200` on `GET http://addr:11434/` as
the Ollama liveness check (§1.5). Real Ollama answers `200 "Ollama is running"`; the test fake just
writes a 200 with an empty body (`services/tests/broker_management_test.go:43`).

**NDJSON streaming** is handled the same way as SSE — it is not parsed. The proxy's own test fixture
writes newline-delimited JSON chunks with an explicit `Flush()` after each
(`services/ollama-proxy/zombie_test.go:118-124`, `:330-340`), and the proxy relays them byte-for-byte.
Because the upstream sends no `Content-Length`, Go's `ReverseProxy` flushes per write.

### 3.3 Model-name matching

**OpenAI lane — exact, case-sensitive string equality, no normalization:**

```go
func nodeAdvertisesModel(n Node, model string) bool {
    if model == "" { return false }
    for _, available := range n.Models {
        if available == model { return true }
    }
    return false
}
```
`services/lmstudio-proxy/proxy.go:1528-1538`

**Ollama lane — `:latest` normalization on both sides, still case-sensitive:**

```go
func ollamaModelKey(model string) string {
    model = strings.TrimSpace(model)
    name := model[strings.LastIndex(model, "/")+1:]
    if name != "" && !strings.ContainsAny(name, ":@") {
        return model + ":latest"
    }
    return model
}
func nodeAdvertisesModel(n Node, model string) bool {
    requested := ollamaModelKey(model)
    if requested == "" { return false }
    for _, available := range n.Models {
        if ollamaModelKey(available) == requested { return true }
    }
    return false
}
```
`services/ollama-proxy/proxy.go:967-974`, `:1731-1742`

Semantics: surrounding whitespace is trimmed; a reference whose **last path segment** contains
neither `:` nor `@` gets `:latest` appended. So `llama` ≡ `llama:latest`, and
`registry.example.com/ns/llama` ≡ `registry.example.com/ns/llama:latest`. **No lowercasing** — `Llama`
and `llama` do not match. The same key function de-duplicates the merged `/api/tags` inventory
(`services/ollama-proxy/proxy.go:1062-1074`), preferring `model` over `name` on each record.

Test: `services/ollama-proxy/failover_test.go:528-565` — a node advertising `llama:latest` is the
sole candidate for a request naming `llama`.

An empty `models` list on a node makes it **ineligible for any model-bearing inference** while
leaving it available for non-inference routes
(`services/ollama-proxy/failover_test.go:577-585`, `services/ollama-proxy/discovery.go:39-43`).

### 3.4 Scheduler ranking among owners

Scheduler side (`services/nvpair-job-scheduler/schedule.go:110-120`) — sort key, ascending:

1. `Pending + GPUPressure`
2. `GPUPressure`
3. `ID`

`Pending` is the count of `queued`/`running` workloads whose `scheduledOn` names that node
(`services/nvpair-job-scheduler/schedule.go:84-98`, `services/nvpair-job-scheduler/state.go:144-153`).
`Rank` is then the index in that order (`services/nvpair-job-scheduler/schedule.go:121-125`).

Wire type (`services/shared/schedulerwire/schedulerwire.go:13-28`):

```go
type NodeRank struct {
    ID          string `json:"id"`
    Pending     int    `json:"pending"`
    GPUPressure int    `json:"gpuPressure"`
    Rank        int    `json:"rank"`
}
type Priority struct {
    Nodes []string   `json:"nodes"`
    Ranks []NodeRank `json:"ranks,omitempty"`
}
```

`MaxGPUPressure = 3` (`services/shared/schedulerwire/schedulerwire.go:10`). Each proxy stores the
snapshot with `Pending` and `GPUPressure` clamped to `[0, 3]` and resets its local reservation
counters (`services/ollama-proxy/proxy.go:2030-2057`). Ranking keys are `hostUuid`, i.e. the same
`id` the broker used in `node/add-manual`
(`services/ollama-proxy/proxy.go:2085-2114`, `services/nvpair-ui-broker/manualnodes.go:168-186`).

---

## 4. Full peer node (later phase)

### 4.1 `_nvpair-node._tcp` TXT format

One mDNS record per node (`services/shared/noderec/noderec.go:9-13`):

```
v=1;uuid=<hostUuid>;cluster-uuid=<clusterUuid>;ip=192.168.1.10;ips=192.168.1.10,10.0.0.2;
ni=14318;ol=11434;lm=1234;er=14319;wl=14320;cl=14321;em=14322;ec=<port>
```

- Service type `_nvpair-node._tcp`, domain `local`
  (`services/shared/noderec/noderec.go:38-42`).
- SRV port is fixed at 14318 and **non-authoritative** — consumers must ignore it and read every
  port from TXT (`services/shared/noderec/noderec.go:43-46`).
- Schema `v=1` (`services/shared/noderec/noderec.go:47-50`); flag-day, not mixed-version.
- Non-port keys: `v`, `uuid`, `cluster-uuid`, `ip`, `ips`
  (`services/shared/noderec/noderec.go:54-66`). `ips` is comma-separated, `ip` first, capped at 4
  entries at both publish and parse (`:68-76`, `:200-212`, `:244-251`), and is emitted only when it
  says something `ip` does not (`:246`).
- Service keys and their ports (`services/shared/noderec/noderec.go:86-103`):
  `ni` node-info, `ol` Ollama, `lm` LM Studio, `er` errors, `wl` workload-manager,
  `cl` cluster-manager, `em` engine-manager LAN HTTP, `ec` engine-manager cluster control.
  **A missing key means that service is absent.**
- Emit order is deterministic: schema, uuid, cluster-uuid, ip, ips, then services in
  `serviceKeyOrder`, then unknown keys sorted (`services/shared/noderec/noderec.go:222-274`,
  `:105-110`).
- Each `key=value` entry must be ≤ 255 bytes (`services/shared/noderec/noderec.go:78-81`, `:276-287`).
- **The model list is NOT in TXT.** It moved to HTTP `em /v1/models`
  (`services/shared/noderec/noderec.go:26-28`).
- Transport is **not advertised** — it is derived from static policy
  (`services/shared/noderec/noderec.go:19-22`, `:112-153`): node-info and the engines are always
  plain; errors/workload/engine-control are mTLS **iff the target is clustered**; cluster-manager is
  the first-byte split.
- Parsing never errors; unknown keys are ignored and malformed ports skipped
  (`services/shared/noderec/noderec.go:180-220`).

### 4.2 Services required for a peer to be routable

From `services/tests/secure_inference_test.go` (the cross-process proof), a peer B receives
inference from peer A only when all of the following hold:

1. **B advertises the engine service key.** `subscribedToNode` requires `Services["ol"]` (or `["lm"]`)
   and a non-empty `IP`, else the node is not a routing target at all
   (`services/ollama-proxy/proxy.go:2085-2089`, `services/lmstudio-proxy/proxy.go:1828-1832`).
   In the test this is pushed as
   `"services":{"ol":{"port":<proxyPort>}}` (`services/tests/secure_inference_test.go:177-190`).
   Note the advertised `ol` port is **B's promoted proxy**, not B's raw engine.
2. **B carries a `hostUuid`** — it becomes the candidate id, `scheduledOn`, and the scheduler key
   (`services/ollama-proxy/proxy.go:2090-2097`).
3. **B carries a `clusterUuid` and A holds a pin for it.** Without a pin, `resolveCandidates` drops
   the peer outright (`services/ollama-proxy/proxy.go:1617-1623`,
   `services/ollama-proxy/ingress_test.go:18-39`). A's dial then uses `https://` to B's `ol` port
   (`services/ollama-proxy/proxy.go:1602-1613`).
4. **B's proxy is running with `--cluster-dir`** so its listener terminates cluster mTLS, and B has
   been told its local engine via `node/set-local-backend {engine, host, port, healthy}`
   (`services/tests/secure_inference_test.go:166-172`, `services/ollama-proxy/ingress.go:26-56`).
   Without a healthy local backend the ingress answers `503 no-local-backend`
   (`services/ollama-proxy/ingress.go:106-111`).
5. **B holds a pin for A.** B's ingress verifies the client cert per request; deleting A's pin
   rejects A on the *next* request with 403, no restart
   (`services/tests/secure_inference_test.go:314-333`, `services/ollama-proxy/ingress.go:94-105`).
6. **Model eligibility** — B must advertise the requested model, carried in
   `modelsByEngine` (`services/tests/secure_inference_test.go:186`,
   `services/shared/noderec/noderec.go:563-578`).

A foreign node that completes the TLS handshake but is not pinned gets a **403**, not a handshake
failure (`services/tests/secure_inference_test.go:300-312`).

`ni` (node-info) is not required for routing, but without it the peer contributes no telemetry and
sits at neutral pressure 1 (§2.4).

### 4.3 `nvpair-cluster-manager` :14321 split listener

- Default port 14321 (`services/nvpair-cluster-manager/manager.go:28-30`), one `net.Listen`
  (`services/nvpair-cluster-manager/httpserver.go:61-66`).
- The split is by first byte: `0x16` (TLS handshake record) → the TLS sub-listener, anything else
  (an ASCII HTTP method byte) → the plain sub-listener. The peeked byte is pushed back so the
  receiving server reads the stream unchanged
  (`services/shared/splitlisten/splitlisten.go:4-8`, `:26-29`, `:56-72`).
- Peek and hand-off are each bounded at 5 s (`services/shared/splitlisten/splitlisten.go:31-41`).
- Two muxes: the **plain** one exposes only `POST /v1/cluster/pairing`; the **mTLS** one exposes
  `/v1/cluster/pairing` plus members-remove and roster
  (`services/nvpair-cluster-manager/httpserver.go:52-59`).
- The same splitter is reused by both inference proxies and by node-info's cluster-gated listener
  (`services/ollama-proxy/proxy.go:749-766`, `services/nvpair-node-info/main.go:458-481`).

### 4.4 `POST /v1/cluster/pairing`

Envelope (`services/nvpair-cluster-manager/httpserver.go:32-46`):

```go
type pairingEnvelope struct {
    InviteID string `json:"inviteId"`
    Phase    string `json:"phase"`
    Msg      string `json:"msg"`       // base64 of the opaque EAP-NOOB blob; "" kicks off completion
    Rejected bool   `json:"rejected,omitempty"`
    Reason   string `json:"reason,omitempty"`
}
```

Handler (`services/nvpair-cluster-manager/httpserver.go:69-128`):

- Non-POST → 405. Body capped at 1 MiB (`:24-28`). Missing `inviteId` → 400. Bad base64 → 400.
- Phases: `initial`, `completion`, `cancel`, `decline`, `fail`, `ack`, `expire`; unknown → 400.
- `ack` **requires** a pinned mTLS client (403 otherwise); `fail` requires it only when the request
  arrived over TLS (`:105-122`).
- `initial` and `completion` run on the plain channel; post-commit `ack`/`fail` on the mTLS channel
  (`:32-35`).
- An explicit joiner refusal is `HTTP 409` with `rejected:true` + `reason`, distinct from a protocol
  failure (`:41-45`).

**EAP-NOOB roles** (`services/nvpair-cluster-manager/pairing.go:152-158`):

- Inviter = EAP-NOOB **Server**, `ServerConfig{Dirs: 2}` (server-to-peer OOB only)
  (`services/nvpair-cluster-manager/pairing.go:140-144`).
- Joiner = EAP-NOOB **Peer**, `PeerConfig{PreferDir: 2}`
  (`services/nvpair-cluster-manager/pairing.go:146-150`).
- The OOB nonce is a 6-digit PIN encoded as a 16-byte big-endian Noob — documented security debt
  (`services/nvpair-cluster-manager/pairing.go:120-138`).
- Session state (`inviteId` → crypto state) persists across the separate HTTP requests of the
  Initial Exchange, the human PIN step, and the Completion Exchange
  (`services/nvpair-cluster-manager/pairing.go:160-173`).

### 4.5 Certificate requirements

Minted by `generateLeaf` (`services/nvpair-cluster-manager/identity.go:145-186`):

| Property | Value |
| --- | --- |
| Key type | **Ed25519** (`identity.go:149`); a non-Ed25519 key is rejected on load (`identity.go:129-131`) |
| Self-signed | yes — `CreateCertificate(rand, tmpl, tmpl, pub, priv)` (`identity.go:175`) |
| Subject CN | the node UUID (`identity.go:164`) |
| URI SAN | `urn:nvpair:node:<uuid>` (`identity.go:157`, `services/shared/clustertrust/clustertrust.go:36-39`) |
| DNS SAN | the hostname, display-only, optional (`identity.go:172-174`) |
| Validity | now−1 h to now+2 years (`identity.go:165-166`) |
| KeyUsage | DigitalSignature (`identity.go:167`) |
| ExtKeyUsage | ServerAuth **and** ClientAuth (`identity.go:168`) |
| Serial | random 128-bit (`identity.go:153-156`) |
| Encoding on disk | `node.crt` / `node.key`, PEM, key as PKCS#8 (`identity.go:179-185`, `services/shared/clustertrust/clustertrust.go:51-60`) |

**Pinning is byte-for-byte DER equality; the CA chain is irrelevant.**

- Server side: `ClientAuth: tls.RequireAnyClientCert`, `MinVersion: TLS1.2`; the pin check happens
  per-request in the handler so a non-pinned client receives a real HTTP **403**
  (`services/shared/clustertrust/clustertrust.go:202-214`).
- `VerifyClientPin` extracts the UUID from the client cert's URI SAN (preferred) or CN and requires
  `bytes.Equal(pinned, cert.Raw)` (`services/shared/clustertrust/clustertrust.go:216-232`,
  `:174-179`, `:253-266`).
- Client side: `InsecureSkipVerify: true` plus a `VerifyPeerCertificate` that requires
  `bytes.Equal(rawCerts[0], pinnedPeerDER)` (`services/shared/clustertrust/clustertrust.go:234-251`).
- Pins live at `<clusterDir>/trusted/<peerUUID>.json` and are reloaded per request; a file whose
  inner `nodeUuid`, filename, and embedded cert disagree is rejected
  (`services/shared/clustertrust/clustertrust.go:103-172`,
  `services/tests/secure_inference_test.go:316-322`).
- Fingerprints used in the pairing protocol are `"sha256:" + hex(SHA-256(DER))`
  (`services/nvpair-cluster-manager/identity.go:188-201`).

Because a mutual pin is exactly cluster membership, "can complete the mTLS handshake" is the whole
authorization decision — no separate membership check exists
(`services/shared/clustertrust/clustertrust.go:10-16`).

---

## 5. Gotchas & test fixtures

### 5.1 Exact JSON bodies the PAIR test fakes return

Mirror these in the Rust test suite.

**Fake Ollama on 127.0.0.1:11434** (`services/tests/broker_management_test.go:36-51`):

| Route | Status | Body |
| --- | --- | --- |
| `GET /` | 200 | *(empty)* |
| `GET /api/tags` | 200, `application/json` | `{"models":[{"name":"llama3.2:latest"}]}` |

**Fake LM Studio on 127.0.0.1:1234** (`services/tests/broker_management_test.go:57-70`):

| Route | Status | Body |
| --- | --- | --- |
| `GET /v1/models` | 200, `application/json` | `{"object":"list","data":[{"id":"qwen2.5-7b-instruct","object":"model"}]}` |

**Fake node-info on 127.0.0.1:14318** (`services/tests/broker_management_test.go:93-111`):

| Route | Status | Body |
| --- | --- | --- |
| `GET /v1/node-info` | 200, `application/json` | `{"GPUs":[],"hostUuid":"learned-host-uuid"}` |

This is the **minimum viable node-info body**: an empty GPU array plus a `hostUuid`. `telemetryValid`
and `msSince` default to `false`/`0` on decode.

**Fake Ollama backend for the secure-inference path** (`services/tests/secure_inference_test.go:214-237`):

| Route | Status | Body |
| --- | --- | --- |
| `GET /api/tags` | 200 | `{"models":[{"name":"m:latest","model":"m:latest"}]}` |
| `POST /api/generate` | 200 | `{"model":"m:latest","response":"hello from the backend","done":true}` |
| anything else | 404 | *(empty)* |

Request body used against it: `{"model":"m:latest","prompt":"hi","stream":false}`
(`services/tests/secure_inference_test.go:279`).

**Ghost-node fake node-info** (`services/tests/ghost_node_test.go:52-77`):
`GET /v1/node-info` → `{"GPUs":[],"hostUuid":"<uuid>"}`; any other path → 404.

**Routing-interop upstream** (`services/tests/model_routing_interop_test.go:23-39`) — used for the
404-then-200 failover proof:

| Configured status | Body |
| --- | --- |
| 200 | `{"done":true,"choices":[]}` |
| non-200 (404 in the test) | `{"error":"model not found"}` |

Request: `POST /api/chat` (Ollama) or `POST /v1/chat/completions` (OpenAI) with
`{"model":"strict-routing-<lane>","messages":[]}`
(`services/tests/model_routing_interop_test.go:117-119`).

**Proxy-unit upstreams** (`services/ollama-proxy/failover_test.go`) — the minimal bodies the failover
tests exercise:

| Scenario | Status | Body | Line |
| --- | --- | --- | --- |
| happy path | 200 | `{"done":true}` | `:138`, `:166`, `:193` |
| client error, no failover | 400 | `{"error":"bad request"}` | `:222` |
| busy, failover | 503 | `{"error":"loading model"}` | `:267` |
| model missing, failover on inference only | 404 | `{"error":"model not found"}` | `:333` |
| `/api/tags` fan-out A | 200 | `{"models":[{"name":"a","model":"a","digest":"a-only"},{"name":"shared","digest":"first"}]}` | `:383` |
| `/api/tags` fan-out B | 200 | `{"models":[{"name":"shared:latest","model":"shared:latest","digest":"second"},{"name":"c","model":"c","digest":"c-only"}]}` | `:384` |
| malformed fan-out member | 200 | `{"models":null}` | `:387` |
| `/v1/models` fan-out A | 200 | `{"object":"list","data":[{"id":"a","owned_by":"a"},{"id":"shared","owned_by":"first"}]}` | `:459` |
| `/v1/models` fan-out B | 200 | `{"object":"list","data":[{"id":"shared","owned_by":"second"},{"id":"c","owned_by":"b"}]}` | `:466` |
| empty inventory | 200 | `{"models":[]}` | `:498`, `:512` |
| streaming chunk | 200 | `{"response":"first chunk","done":false}\n` (flushed) | `services/ollama-proxy/zombie_test.go:121` |
| partial stream, client vanishes | 200 | `{"model":"llama","response":"partial tokens before the client vanished","done":false}` | `services/ollama-proxy/zombie_test.go:72` |

Node fixture used by all of them (`services/ollama-proxy/failover_test.go:32-54`):
`Node{ID: id, Addresses: []string{host}, Port: port}`, optionally with `Models: []string{model}`.
Note `Host` and `TXT` are left empty — `nodeCandidates` falls back to `Addresses`
(`services/ollama-proxy/proxy.go:1858-1894`).

**Manual-nodes unit fixtures** (`services/nvpair-manual-nodes/manager_test.go:86-134`) confirm the
exact host:port/path triples the prober issues:
`GET <addr>:11434/`, `GET <addr>:11434/api/tags`, `GET <addr>:14318/v1/node-info`,
`GET <addr>:1234/v1/models`.

### 5.2 403 for non-loopback plaintext on proxy ports

Both proxies refuse plaintext from anything but loopback
(`services/ollama-proxy/ingress.go:63-85`, `services/lmstudio-proxy/ingress.go:63-85`):

```json
{"error":"plaintext requests are accepted only from loopback; cluster peers must use the mTLS ingress",
 "code":"loopback-only"}
```

`HTTP 403`, `Content-Type: application/json`, `X-Content-Type-Options: nosniff`, **plus CORS
headers** — a browser must be able to read the reason
(`services/ollama-proxy/ingress.go:151-165`,
`services/ollama-proxy/ingress_test.go:44-59`,
`services/lmstudio-proxy/ingress_test.go:39-54`).

Loopback = `127.0.0.0/8` or `::1`; an unparseable or empty `RemoteAddr` fails **closed**
(`services/ollama-proxy/ingress.go:139-149`, `services/ollama-proxy/ingress_test.go:107-123`).

Related gotchas on the same handler:

- A **non-loopback OPTIONS preflight is still answered `204`** with permissive CORS, before the
  loopback gate. It grants nothing — the real request that follows still gets 403
  (`services/ollama-proxy/ingress.go:65-71`, `services/ollama-proxy/ingress_test.go:61-78`).
- A loopback request carrying `X-NVPAIR-Engine-Identity-Probe: 1` gets **409**
  `{"error":"the compatibility facade is not an LM Studio engine","code":"proxy-facade"}` — the
  facade must never satisfy an engine readiness probe
  (`services/ollama-proxy/ingress.go:78-83`, `services/lmstudio-proxy/ingress.go:78-83`).
- The mTLS ingress on an unclustered node rejects **everything** with 403 — no pins exist
  (`services/ollama-proxy/ingress_test.go:93-105`).

### 5.3 Other gotchas for an implementer

1. **`GPUs` is PascalCase.** Every other key in `/v1/node-info` is snake_case or camelCase
   (`services/nvpair-node-info/main.go:75`).
2. **`msSince`/`telemetryValid` are camelCase**, while `vram_bytes`/`total_bytes` are snake_case, in
   the same object (`services/nvpair-node-info/main.go:74-79`).
3. **Zero is indistinguishable from unknown** for every `omitempty` numeric field. This is
   deliberate and documented as benign (`services/nvpair-node-info/main.go:51-58`,
   `services/nvpair-node-info/stats.go:12-18`).
4. **Any telemetry change fires a `node/updated`** — including a 41 % → 42 % utilization tick
   (`services/nvpair-manual-nodes/manager.go:731-739`). Don't jitter utilization needlessly.
5. **Probes disable HTTP keep-alives** (`services/nvpair-manual-nodes/manager.go:196-204`), so every
   10 s the node sees a fresh TCP connection to each of 11434 / 1234 / 14318.
6. **A manual node dialed by the proxy is plain HTTP only** — never TLS, regardless of the entry's
   `tls_port`, which affects only the node-info probe
   (`services/ollama-proxy/proxy.go:1614-1616`).
7. **A manual node is exempt from the pin requirement**, which is exactly why it is the shortest path
   to being routable from an unclustered PAIR install
   (`services/ollama-proxy/ingress_test.go:13-39`).
8. **The proxy dedups candidates by resolved host:port** — the same machine reachable under a manual
   id and a discovered id is dialed once (`services/ollama-proxy/proxy.go:1631-1634`).
9. **Address confirmation is a bare TCP handshake**, cached per node, 1 s per candidate
   (`services/shared/reach/reach.go:14-20`, `:38-46`,
   `services/ollama-proxy/proxy.go:1840-1847`). A failed forward forgets the cached address
   (`services/ollama-proxy/proxy.go:1412`).
10. **Local-interface addresses are rewritten to `127.0.0.1`** and floated to the front of the
    candidate list (`services/ollama-proxy/proxy.go:1875-1893`).
11. **The `/v1/models` and `/api/tags` fan-out caps a response at 16 MiB** and rejects any record
    without an identity (`id`, or `model`/`name` for `/api/tags`)
    (`services/ollama-proxy/proxy.go:1030-1033`, `:1062-1073`). One bad record fails that whole
    candidate, not the request.
12. **Conflicting digests across candidates are logged, first wins**
    (`services/ollama-proxy/proxy.go:1092-1099`).
13. **Manual-node worker restarts lose all entries** (§1.8) — a client must re-add.
14. The prober's LM Studio port (1234) is also `lmstudio-proxy`'s own default listen port
    (`services/lmstudio-proxy/main.go:22`). On the same host, the proxy's self-target guard prevents
    a loop (`services/lmstudio-proxy/proxy.go:1422-1427`); on a remote Android node this is not a
    concern, but a node that exposes an OpenAI API on 1234 will be probed as LM Studio.
