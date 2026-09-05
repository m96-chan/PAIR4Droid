package com.pair4droid.ui

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.pair4droid.repo.NodeRepository
import com.pair4droid.repo.NodeUiState
import com.pair4droid.util.formatBytes
import java.io.File
import java.io.FileOutputStream
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import uniffi.pair_ffi.ModelInfo

/**
 * Model import/rename/delete (ticket #17). Filesystem operations act directly on the files
 * under [NodeRepository.modelsDir] (unambiguous); [NodeUiState.models] — from
 * `PairNode.listModels()` — is only consulted best-effort for size/quant display, since the
 * exact shape of the generated `ModelInfo` isn't fixed yet (see android/README.md).
 */
@Composable
fun ModelsScreen(state: NodeUiState, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var files by remember { mutableStateOf(listModelFiles()) }
    var importProgress by remember { mutableStateOf<Float?>(null) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var renameTarget by remember { mutableStateOf<File?>(null) }
    var deleteTarget by remember { mutableStateOf<File?>(null) }

    fun refreshFiles() {
        files = listModelFiles()
    }

    val importLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        errorMessage = null
        scope.launch {
            importProgress = 0f
            val result = importModel(context, uri) { progress -> importProgress = progress }
            importProgress = null
            result.exceptionOrNull()?.let { errorMessage = it.message ?: "Import failed" }
            NodeRepository.applyModelChanges()
            refreshFiles()
        }
    }

    Column(
        modifier = modifier.fillMaxSize().padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Models", style = MaterialTheme.typography.headlineSmall)
            Button(onClick = { importLauncher.launch(arrayOf("*/*")) }) { Text("Import .gguf") }
        }

        importProgress?.let { p ->
            LinearProgressIndicator(progress = { p }, modifier = Modifier.fillMaxWidth())
        }
        errorMessage?.let { msg ->
            Text(msg, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
        }

        if (files.isEmpty()) {
            Text("No models imported yet.")
        } else {
            LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(files, key = { it.absolutePath }) { file ->
                    val info = state.models.firstOrNull { it.name == file.nameWithoutExtension || it.name == file.name }
                    ModelRow(
                        file = file,
                        info = info,
                        onRename = { renameTarget = file },
                        onDelete = { deleteTarget = file },
                    )
                }
            }
        }
    }

    renameTarget?.let { file ->
        RenameDialog(
            currentName = file.nameWithoutExtension,
            onDismiss = { renameTarget = null },
            onConfirm = { newName ->
                renameTarget = null
                scope.launch {
                    withContext(Dispatchers.IO) {
                        file.renameTo(File(file.parentFile, "$newName.gguf"))
                    }
                    NodeRepository.applyModelChanges()
                    refreshFiles()
                }
            },
        )
    }

    deleteTarget?.let { file ->
        AlertDialog(
            onDismissRequest = { deleteTarget = null },
            title = { Text("Delete ${file.name}?") },
            text = { Text("This removes the model file from the device. This cannot be undone.") },
            confirmButton = {
                TextButton(onClick = {
                    deleteTarget = null
                    scope.launch {
                        withContext(Dispatchers.IO) { file.delete() }
                        NodeRepository.applyModelChanges()
                        refreshFiles()
                    }
                }) { Text("Delete") }
            },
            dismissButton = { TextButton(onClick = { deleteTarget = null }) { Text("Cancel") } },
        )
    }
}

@Composable
private fun ModelRow(
    file: File,
    info: ModelInfo?,
    onRename: () -> Unit,
    onDelete: () -> Unit,
) {
    val sizeBytes = info?.sizeBytes?.toLong() ?: file.length()
    val quant = info?.quant ?: "unknown"
    val estimatedRam = (sizeBytes * 1.2).toLong()

    ElevatedCard(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(file.nameWithoutExtension, style = MaterialTheme.typography.titleMedium)
            Text("Size: ${formatBytes(sizeBytes)}    Quant: $quant")
            Text(
                "Estimated RAM while loaded: ${formatBytes(estimatedRam)}",
                style = MaterialTheme.typography.bodySmall,
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                TextButton(onClick = onRename) { Text("Rename") }
                TextButton(onClick = onDelete) { Text("Delete") }
            }
        }
    }
}

@Composable
private fun RenameDialog(
    currentName: String,
    onDismiss: () -> Unit,
    onConfirm: (String) -> Unit,
) {
    var text by remember { mutableStateOf(currentName) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Rename model") },
        text = {
            Column {
                Text(
                    "This name is PAIR's exact-match routing key across your fleet — keep it " +
                        "unique, e.g. \"phone-$currentName\".",
                    style = MaterialTheme.typography.bodySmall,
                )
                Spacer(Modifier.height(8.dp))
                OutlinedTextField(value = text, onValueChange = { text = it }, singleLine = true)
            }
        },
        confirmButton = {
            TextButton(onClick = { if (text.isNotBlank()) onConfirm(text) }) { Text("Rename") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

private fun listModelFiles(): List<File> =
    NodeRepository.modelsDir()
        .listFiles { f -> f.isFile && f.extension.equals("gguf", ignoreCase = true) }
        ?.sortedBy { it.name }
        ?: emptyList()

private suspend fun importModel(
    context: Context,
    uri: Uri,
    onProgress: (Float) -> Unit,
): Result<Unit> = withContext(Dispatchers.IO) {
    runCatching {
        val resolver = context.contentResolver
        var displayName = uri.lastPathSegment ?: "model.gguf"
        var size = -1L
        resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val nameIdx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                val sizeIdx = cursor.getColumnIndex(OpenableColumns.SIZE)
                if (nameIdx >= 0) cursor.getString(nameIdx)?.let { displayName = it }
                if (sizeIdx >= 0) size = cursor.getLong(sizeIdx)
            }
        }
        require(displayName.endsWith(".gguf", ignoreCase = true)) {
            "\"$displayName\" is not a .gguf file"
        }
        val target = File(NodeRepository.modelsDir(), displayName)
        val input = resolver.openInputStream(uri) ?: error("Could not open \"$displayName\"")
        input.use { stream ->
            FileOutputStream(target).use { output ->
                val buffer = ByteArray(1 shl 16)
                var copied = 0L
                while (true) {
                    val read = stream.read(buffer)
                    if (read < 0) break
                    output.write(buffer, 0, read)
                    copied += read
                    if (size > 0) onProgress((copied.toFloat() / size).coerceIn(0f, 1f))
                }
            }
        }
    }
}
