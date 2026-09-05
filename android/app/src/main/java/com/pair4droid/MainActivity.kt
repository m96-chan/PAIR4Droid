package com.pair4droid

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.pair4droid.service.NodeService
import com.pair4droid.ui.AppRoot
import com.pair4droid.ui.theme.PAIR4DroidTheme

class MainActivity : ComponentActivity() {

    private val requestNotificationPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { /* either way, proceed */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            requestNotificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        }

        setContent {
            PAIR4DroidTheme {
                AppRoot(
                    onStartNode = {
                        startService(Intent(this, NodeService::class.java).setAction(NodeService.ACTION_START))
                    },
                    onStopNode = {
                        startService(Intent(this, NodeService::class.java).setAction(NodeService.ACTION_STOP))
                    },
                )
            }
        }
    }
}
