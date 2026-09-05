package com.pair4droid.ui

import androidx.lifecycle.ViewModel
import com.pair4droid.repo.NodeRepository
import com.pair4droid.repo.NodeUiState
import kotlinx.coroutines.flow.StateFlow

/**
 * Thin pass-through over [NodeRepository]'s singleton state: the node itself is owned by
 * [com.pair4droid.service.NodeService], not by this ViewModel, so start/stop go through
 * service Intents (see MainActivity) rather than through view-model functions.
 */
class NodeViewModel : ViewModel() {
    val state: StateFlow<NodeUiState> = NodeRepository.state
}
