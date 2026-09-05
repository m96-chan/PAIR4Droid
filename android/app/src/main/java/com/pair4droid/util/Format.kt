package com.pair4droid.util

/** Renders a byte count as e.g. "3.2 GiB" or "480 MiB", switching units at 1 GiB. */
fun formatBytes(bytes: Long): String {
    val gib = bytes / (1024.0 * 1024.0 * 1024.0)
    return if (gib >= 1) "%.1f GiB".format(gib) else "%d MiB".format(bytes / (1024 * 1024))
}
