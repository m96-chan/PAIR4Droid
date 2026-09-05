package uniffi.pair_ffi

/**
 * `object PairNode` — the shape the app codes against (`NodeRepository`, `NodeService`).
 *
 * UniFFI cannot generate a Kotlin `object` with static-looking methods, so
 * `core/crates/pair-ffi/src/lib.rs` exports top-level functions (`pair_node_start`,
 * `pair_node_stop`, …) which UniFFI renders as `pairNodeStart`, `pairNodeStop`, … in this
 * same package (`uniffi.pair_ffi`, generated into `build/generated/uniffi`). This file is
 * the only hand-written Kotlin in that package: a thin, allocation-free facade over them.
 *
 * Every call blocks the calling thread — the Rust side owns a dedicated tokio runtime and
 * `block_on`s it — so call these from `Dispatchers.IO`, as `NodeRepository` does.
 */
object PairNode {

    /** Binds the three lanes and starts serving. Throws [PairException.AlreadyRunning] if up. */
    @Throws(PairException::class)
    fun start(config: NodeConfig): NodeStatus = pairNodeStart(config)

    /** Stops the lanes and frees the ports. Throws [PairException.NotRunning] if stopped. */
    @Throws(PairException::class)
    fun stop() = pairNodeStop()

    /** Current status; valid whether or not the node is running. */
    fun status(): NodeStatus = pairNodeStatus()

    /** Pushes battery/thermal/screen signals into telemetry (Rust never polls Android APIs). */
    fun pushSignals(signals: ExternalSignals) = pairNodePushSignals(signals)

    /** Sets the directory holding `*.gguf` files. Takes effect on the next [start]. */
    fun setModelsDir(path: String) = pairNodeSetModelsDir(path)

    /** The running engine's catalogue, or a scan of the models dir while stopped. */
    fun listModels(): List<ModelInfo> = pairNodeListModels()

    /** Registers the sink for log / request / state-change events. Replaces any previous one. */
    fun setEventListener(events: NodeEvents) = pairNodeSetEventListener(events)
}
