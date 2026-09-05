package com.pair4droid.repo

import android.content.Context
import android.os.Build
import android.util.Log
import com.pair4droid.BuildConfig
import java.io.File
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.withContext
import uniffi.pair_ffi.ExternalSignals
import uniffi.pair_ffi.ModelInfo
import uniffi.pair_ffi.NodeConfig
import uniffi.pair_ffi.NodeEvents
import uniffi.pair_ffi.NodeStatus
import uniffi.pair_ffi.PairNode

/**
 * Ports PAIR expects the node on — compile-time constants on the Rust side
 * (docs/architecture.md, CLAUDE.md "Design invariants"). Mirrored here to fill [NodeConfig]
 * and to render the UI before the node has actually started.
 */
const val OPENAI_PORT = 1234
const val OLLAMA_PORT = 11434
const val NODE_INFO_PORT = 14318

private const val PREFS_NAME = "pair4droid"
private const val PREF_HOST_UUID = "host_uuid"
private const val MAX_LOG_LINES = 50
private const val TAG = "NodeRepository"

data class NodeUiState(
    val running: Boolean = false,
    val wifiIpv4: String? = null,
    val loadedModel: String? = null,
    val active: Int = 0,
    val queued: Int = 0,
    val lastError: String? = null,
    val models: List<ModelInfo> = emptyList(),
    val recentLog: List<String> = emptyList(),
)

/**
 * Owns the single [PairNode] instance for the process and exposes its state as a
 * [StateFlow] so the Compose UI and [com.pair4droid.service.NodeService] see the same
 * picture. Per pair-ffi/src/lib.rs, every FFI call blocks on the Rust side ("Kotlin wraps
 * them in coroutines"), so every call here is dispatched onto [Dispatchers.IO].
 */
object NodeRepository {

    private lateinit var appContext: Context

    private val _state = MutableStateFlow(NodeUiState())
    val state: StateFlow<NodeUiState> = _state.asStateFlow()

    private val events = object : NodeEvents {
        override fun onLog(level: String, msg: String) {
            appendLog("[$level] $msg")
        }

        override fun onRequest(lane: String, model: String, status: Int, ms: Long) {
            appendLog("$lane $model -> $status (${ms}ms)")
        }

        override fun onStateChanged(status: NodeStatus) {
            applyStatus(status)
        }
    }

    fun init(context: Context) {
        if (::appContext.isInitialized) return
        appContext = context.applicationContext

        // ASSUMPTION (flag for the design session): the doc comment in pair-ffi/src/lib.rs
        // lists `NodeEvents` as a callback interface Kotlin implements but does not name how
        // it gets registered with `PairNode` (start(config) alone is documented, with no
        // events parameter). This guesses a `setEventListener` entry point; if pair-ffi wires
        // events through `start(config, events)` instead, move this call there.
        runCatching {
            PairNode.setEventListener(events)
        }.onFailure {
            Log.w(TAG, "PairNode.setEventListener unavailable; event log/state pushes will not arrive", it)
        }
    }

    fun modelsDir(): File = File(appContext.filesDir, "models").apply { mkdirs() }

    fun hostUuid(): String {
        val prefs = appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(PREF_HOST_UUID, null) ?: UUID.randomUUID().toString().also {
            prefs.edit().putString(PREF_HOST_UUID, it).apply()
        }
    }

    private fun acceleratorName(): String =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) Build.SOC_MODEL else Build.HARDWARE

    /** Starts the node, sizing its model budget from [recommendedMaxModelBytes]. */
    suspend fun start(modelBudgetBytes: Long): NodeStatus = withContext(Dispatchers.IO) {
        val ggufFiles = modelsDir().listFiles { f -> f.isFile && f.extension.equals("gguf", ignoreCase = true) }
        val mockModels = if (ggufFiles.isNullOrEmpty() && BuildConfig.MOCK_MODELS.isNotBlank()) {
            listOf(BuildConfig.MOCK_MODELS)
        } else {
            emptyList()
        }

        val config = NodeConfig(
            bind = "0.0.0.0",
            openaiPort = OPENAI_PORT.toUShort(),
            ollamaPort = OLLAMA_PORT.toUShort(),
            nodeInfoPort = NODE_INFO_PORT.toUShort(),
            hostUuid = hostUuid(),
            acceleratorName = acceleratorName(),
            modelBudgetBytes = modelBudgetBytes.toULong(),
            mockModels = mockModels,
        )

        PairNode.setModelsDir(modelsDir().absolutePath)
        val status = PairNode.start(config)
        applyStatus(status)
        refreshModelsLocked()
        status
    }

    suspend fun stop() = withContext(Dispatchers.IO) {
        PairNode.stop()
        applyStatus(PairNode.status())
    }

    suspend fun refreshStatus() = withContext(Dispatchers.IO) {
        applyStatus(PairNode.status())
    }

    suspend fun refreshModels() = withContext(Dispatchers.IO) { refreshModelsLocked() }

    private fun refreshModelsLocked() {
        val models = PairNode.listModels()
        _state.update { it.copy(models = models) }
    }

    /** Re-scans [modelsDir] after an import/rename/delete (ticket #17: "restart... automatically"). */
    suspend fun applyModelChanges() = withContext(Dispatchers.IO) {
        PairNode.setModelsDir(modelsDir().absolutePath)
        refreshModelsLocked()
    }

    suspend fun pushSignals(signals: ExternalSignals) = withContext(Dispatchers.IO) {
        PairNode.pushSignals(signals)
    }

    fun setWifiIpv4(ip: String?) {
        _state.update { it.copy(wifiIpv4 = ip) }
    }

    private fun applyStatus(status: NodeStatus) {
        _state.update {
            it.copy(
                running = status.running,
                loadedModel = status.loadedModel,
                active = status.active.toInt(),
                queued = status.queued.toInt(),
                lastError = status.lastError,
            )
        }
    }

    private fun appendLog(line: String) {
        _state.update { it.copy(recentLog = (it.recentLog + line).takeLast(MAX_LOG_LINES)) }
    }
}
