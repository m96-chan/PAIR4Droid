# ADR-0001: Implement PAIR's manual-node contract first, full peer later

**Status:** accepted · 2026-09-05

## Context
PAIR has two participation paths. A *full peer* advertises `_nvpair-node._tcp`, listens on 8
ports, runs pin-based mTLS and EAP-NOOB pairing (`services/nvpair-cluster-manager`). A *manual
node* is an IP the user types in; PAIR probes fixed plain-HTTP ports (`services/nvpair-manual-nodes/manager.go:250-300`)
and PAIR's own interop test registers a bare `httptest.Server` as one (`services/tests/model_routing_interop_test.go:88-108`).

## Decision
Phase 1 implements only the manual-node surface: `:1234` OpenAI lane, `:11434` Ollama lane,
`:14318` node-info. Zero PAIR modifications.

## Consequences
+ Works against unmodified PAIR today; small, fully testable in Rust on a Linux host.
+ Both lanes are implemented so the user can pick either in PAIR's UI (the OpenAI lane has the
  looser probe; the Ollama lane needs `GET /` → 200).
− No auto-discovery; the user types the phone's IP. Phone IP changes need re-adding.
− No mTLS; the lanes are plaintext on the LAN (same as any manual node PAIR supports).
