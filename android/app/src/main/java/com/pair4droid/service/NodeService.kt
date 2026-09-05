package com.pair4droid.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ServiceInfo
import android.net.ConnectivityManager
import android.net.LinkProperties
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.net.wifi.WifiManager
import android.os.BatteryManager
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.lifecycle.LifecycleService
import androidx.lifecycle.lifecycleScope
import com.pair4droid.MainActivity
import com.pair4droid.R
import com.pair4droid.repo.NODE_INFO_PORT
import com.pair4droid.repo.NodeRepository
import com.pair4droid.repo.OLLAMA_PORT
import com.pair4droid.repo.OPENAI_PORT
import com.pair4droid.util.androidThermalStatusToFfi
import com.pair4droid.util.recommendedMaxModelBytes
import com.pair4droid.util.wifiIpv4Address
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import uniffi.pair_ffi.ExternalSignals

/**
 * Foreground service (`specialUse`) that owns the [PairNode][uniffi.pair_ffi.PairNode]
 * lifecycle: starts/stops it, holds the Wi-Fi + wake locks while it runs, and pushes
 * battery/thermal/screen signals into Rust on every change and at least every 30s
 * (docs/architecture.md "Android lifecycle").
 */
class NodeService : LifecycleService() {

    private var wifiLock: WifiManager.WifiLock? = null
    private var wakeLock: PowerManager.WakeLock? = null

    private var batteryPercent: Int = 100
    private var charging: Boolean = false
    private var screenOn: Boolean = true
    private var thermalStatus: Int = PowerManager.THERMAL_STATUS_NONE

    private val batteryReceiver = object : BroadcastReceiver() {
        override fun onReceive(receiverContext: Context, intent: Intent) {
            val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
            val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
            if (level >= 0 && scale > 0) {
                batteryPercent = (level * 100) / scale
            }
            val status = intent.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
            charging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                status == BatteryManager.BATTERY_STATUS_FULL
            pushSignalsNow()
        }
    }

    private val screenReceiver = object : BroadcastReceiver() {
        override fun onReceive(receiverContext: Context, intent: Intent) {
            screenOn = intent.action == Intent.ACTION_SCREEN_ON
            pushSignalsNow()
        }
    }

    private val thermalListener = PowerManager.OnThermalStatusChangedListener { status ->
        thermalStatus = status
        pushSignalsNow()
    }

    private val networkCallback = object : ConnectivityManager.NetworkCallback() {
        override fun onAvailable(network: Network) = refreshIp()
        override fun onLost(network: Network) = refreshIp()
        override fun onLinkPropertiesChanged(network: Network, linkProperties: LinkProperties) = refreshIp()
    }

    override fun onCreate() {
        super.onCreate()

        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                getString(R.string.notification_channel_name),
                NotificationManager.IMPORTANCE_LOW,
            ),
        )

        registerReceiver(batteryReceiver, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        registerReceiver(
            screenReceiver,
            IntentFilter(Intent.ACTION_SCREEN_ON).apply { addAction(Intent.ACTION_SCREEN_OFF) },
        )

        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        powerManager.addThermalStatusListener(mainExecutor, thermalListener)

        val connectivityManager = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        connectivityManager.registerNetworkCallback(
            NetworkRequest.Builder().addTransportType(NetworkCapabilities.TRANSPORT_WIFI).build(),
            networkCallback,
        )

        refreshIp()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        if (intent?.action == ACTION_STOP) {
            stopNode()
            return START_NOT_STICKY
        }
        startNode()
        return START_STICKY
    }

    private fun startNode() {
        startForegroundWithNotification("Starting…")
        acquireLocks()

        lifecycleScope.launch(Dispatchers.IO) {
            val budget = recommendedMaxModelBytes(totalMemBytes())
            NodeRepository.start(budget)
            pushSignalsNow()
        }
        updateNotification()

        lifecycleScope.launch {
            while (isActive) {
                delay(30_000)
                pushSignalsNow()
                // Belt-and-suspenders alongside NodeEvents.onStateChanged (see
                // NodeRepository.init's ASSUMPTION comment): poll status directly too, in
                // case that callback wiring isn't what pair-ffi ends up exposing.
                NodeRepository.refreshStatus()
                updateNotification()
            }
        }
    }

    private fun stopNode() {
        lifecycleScope.launch(Dispatchers.IO) { NodeRepository.stop() }
        releaseLocks()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun pushSignalsNow() {
        lifecycleScope.launch(Dispatchers.IO) {
            NodeRepository.pushSignals(
                ExternalSignals(
                    batteryPercent = batteryPercent.coerceIn(0, 100).toUByte(),
                    charging = charging,
                    thermal = androidThermalStatusToFfi(thermalStatus),
                    screenOn = screenOn,
                ),
            )
        }
    }

    private fun refreshIp() {
        NodeRepository.setWifiIpv4(wifiIpv4Address(this))
        updateNotification()
    }

    private fun totalMemBytes(): Long {
        val activityManager = getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val info = android.app.ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(info)
        return info.totalMem
    }

    private fun acquireLocks() {
        val wifiManager = getSystemService(Context.WIFI_SERVICE) as WifiManager
        wifiLock = wifiManager.createWifiLock(WifiManager.WIFI_MODE_FULL_LOW_LATENCY, "pair4droid:node").apply {
            setReferenceCounted(false)
            acquire()
        }
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "pair4droid:node").apply {
            setReferenceCounted(false)
            acquire()
        }
    }

    private fun releaseLocks() {
        wifiLock?.let { if (it.isHeld) it.release() }
        wifiLock = null
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
    }

    private fun startForegroundWithNotification(overrideState: String) {
        ServiceCompat.startForeground(
            this,
            NOTIFICATION_ID,
            buildNotification(overrideState),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE,
        )
    }

    private fun updateNotification() {
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(NOTIFICATION_ID, buildNotification(null))
    }

    private fun buildNotification(overrideState: String?): Notification {
        val text = overrideState ?: run {
            val s = NodeRepository.state.value
            val ip = s.wifiIpv4 ?: "no Wi-Fi"
            "$ip  ·  $OPENAI_PORT/$OLLAMA_PORT/$NODE_INFO_PORT  ·  ${if (s.running) "running" else "stopped"}"
        }

        val contentIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val stopIntent = PendingIntent.getService(
            this,
            0,
            Intent(this, NodeService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE,
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(text)
            .setContentIntent(contentIntent)
            .addAction(0, "Stop", stopIntent)
            .setOngoing(true)
            .build()
    }

    override fun onDestroy() {
        releaseLocks()
        runCatching { unregisterReceiver(batteryReceiver) }
        runCatching { unregisterReceiver(screenReceiver) }
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        runCatching { powerManager.removeThermalStatusListener(thermalListener) }
        val connectivityManager = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        runCatching { connectivityManager.unregisterNetworkCallback(networkCallback) }
        super.onDestroy()
    }

    companion object {
        const val ACTION_START = "com.pair4droid.action.START"
        const val ACTION_STOP = "com.pair4droid.action.STOP"
        private const val CHANNEL_ID = "node_service"
        private const val NOTIFICATION_ID = 1
    }
}
