package com.pair4droid.util

import android.content.Context
import android.net.ConnectivityManager
import android.net.LinkProperties
import android.net.NetworkCapabilities
import java.net.Inet4Address

/**
 * The device's LAN IPv4 address on Wi-Fi, if any — what the user types into PAIR's
 * "add manual node" dialog (docs/architecture.md).
 */
fun wifiIpv4Address(context: Context): String? {
    val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager ?: return null

    val active = cm.activeNetwork
    val activeCaps = active?.let(cm::getNetworkCapabilities)
    if (activeCaps != null && activeCaps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)) {
        ipv4From(cm.getLinkProperties(active))?.let { return it }
    }

    // The active network may be a VPN or ethernet path while Wi-Fi is still up and is what
    // PAIR needs to reach us on, so fall back to scanning all known networks for one.
    for (network in cm.allNetworks) {
        val caps = cm.getNetworkCapabilities(network) ?: continue
        if (caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)) {
            ipv4From(cm.getLinkProperties(network))?.let { return it }
        }
    }
    return null
}

private fun ipv4From(linkProperties: LinkProperties?): String? =
    linkProperties?.linkAddresses
        ?.mapNotNull { it.address as? Inet4Address }
        ?.firstOrNull { !it.isLoopbackAddress }
        ?.hostAddress
