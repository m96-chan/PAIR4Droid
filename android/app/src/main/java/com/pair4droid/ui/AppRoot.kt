package com.pair4droid.ui

import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.pair4droid.R

private const val TAB_NODE = 0
private const val TAB_MODELS = 1
private const val TAB_SETTINGS = 2

/** The app's top-level Compose entry point: a bottom-tab shell over the three screens. */
@Composable
fun AppRoot(
    onStartNode: () -> Unit,
    onStopNode: () -> Unit,
    viewModel: NodeViewModel = viewModel(),
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    var tab by rememberSaveable { mutableStateOf(TAB_NODE) }

    Scaffold(
        topBar = { TopAppBar(title = { Text(stringResource(R.string.app_name)) }) },
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    selected = tab == TAB_NODE,
                    onClick = { tab = TAB_NODE },
                    icon = { Icon(Icons.Filled.Home, contentDescription = null) },
                    label = { Text("Node") },
                )
                NavigationBarItem(
                    selected = tab == TAB_MODELS,
                    onClick = { tab = TAB_MODELS },
                    icon = { Icon(Icons.Filled.Storage, contentDescription = null) },
                    label = { Text("Models") },
                )
                NavigationBarItem(
                    selected = tab == TAB_SETTINGS,
                    onClick = { tab = TAB_SETTINGS },
                    icon = { Icon(Icons.Filled.Settings, contentDescription = null) },
                    label = { Text("Settings") },
                )
            }
        },
    ) { innerPadding ->
        when (tab) {
            TAB_NODE -> NodeScreen(
                state = state,
                onToggle = { checked -> if (checked) onStartNode() else onStopNode() },
                modifier = Modifier.padding(innerPadding),
            )
            TAB_MODELS -> ModelsScreen(state = state, modifier = Modifier.padding(innerPadding))
            else -> SettingsScreen(modifier = Modifier.padding(innerPadding))
        }
    }
}
