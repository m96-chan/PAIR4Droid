package com.pair4droid.ui

import android.app.ActivityManager
import android.content.Context
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.pair4droid.repo.NODE_INFO_PORT
import com.pair4droid.repo.NodeUiState
import com.pair4droid.repo.OLLAMA_PORT
import com.pair4droid.repo.OPENAI_PORT
import com.pair4droid.util.formatBytes
import com.pair4droid.util.recommendedMaxModelBytes

@Composable
fun NodeScreen(
    state: NodeUiState,
    onToggle: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val recommendedBytes = remember { recommendedMaxModelBytes(totalMemBytes(context)) }

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column {
                Text("PAIR node", style = MaterialTheme.typography.headlineSmall)
                Text(
                    if (state.running) "Running" else "Stopped",
                    style = MaterialTheme.typography.bodyMedium,
                    color = if (state.running) {
                        MaterialTheme.colorScheme.primary
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    },
                )
            }
            Switch(checked = state.running, onCheckedChange = onToggle)
        }

        ElevatedCard(modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("Add this node in PAIR", style = MaterialTheme.typography.titleMedium)
                Text("PAIR → Nodes → Add manual node → ${state.wifiIpv4 ?: "waiting for Wi-Fi…"}")
                Spacer(Modifier.height(8.dp))
                PortRow("OpenAI lane", OPENAI_PORT)
                PortRow("Ollama lane", OLLAMA_PORT)
                PortRow("Node info", NODE_INFO_PORT)
            }
        }

        ElevatedCard(modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("Status", style = MaterialTheme.typography.titleMedium)
                Text("Loaded model: ${state.loadedModel ?: "none"}")
                Text("Active requests: ${state.active}    Queued: ${state.queued}")
                if (state.lastError != null) {
                    Text(
                        "Last error: ${state.lastError}",
                        color = MaterialTheme.colorScheme.error,
                    )
                }
            }
        }

        ElevatedCard(modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("Models", style = MaterialTheme.typography.titleMedium)
                if (state.models.isEmpty()) {
                    Text("No models loaded yet — add one from the Models tab.")
                } else {
                    state.models.forEach { model -> Text("• ${model.name}") }
                }
                Text(
                    "Recommended max model size on this device: ${formatBytes(recommendedBytes)} " +
                        "— 60% of what Android keeps free for a not-visible foreground service.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        ElevatedCard(modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("Recent requests", style = MaterialTheme.typography.titleMedium)
                if (state.recentLog.isEmpty()) {
                    Text("Nothing yet.", style = MaterialTheme.typography.bodySmall)
                } else {
                    LazyColumn(modifier = Modifier.heightIn(max = 240.dp)) {
                        items(state.recentLog.asReversed()) { line ->
                            Text(
                                line,
                                style = MaterialTheme.typography.bodySmall,
                                fontFamily = FontFamily.Monospace,
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun PortRow(label: String, port: Int) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label)
        Text(port.toString(), fontFamily = FontFamily.Monospace)
    }
}

private fun totalMemBytes(context: Context): Long {
    val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    val info = ActivityManager.MemoryInfo()
    am.getMemoryInfo(info)
    return info.totalMem
}
