package com.pair4droid.util

private const val GIB = 1L shl 30

/**
 * Android's per-process memory limiter treats a Foreground Service as "not visible" and
 * caps its budget by a step function of total device RAM (CLAUDE.md "Android specifics",
 * docs/architecture.md "Android lifecycle"). This mirrors the known steps.
 *
 * Devices below the lowest documented step (6 GiB) are not in the table; 1 GiB is used as
 * a conservative floor rather than extrapolating.
 */
fun notVisibleCapBytes(totalMemBytes: Long): Long = when {
    totalMemBytes >= 16L * GIB -> 5L * GIB
    totalMemBytes >= 12L * GIB -> 4L * GIB
    totalMemBytes >= 8L * GIB -> 3L * GIB
    totalMemBytes >= 6L * GIB -> 2L * GIB
    else -> 1L * GIB
}

/**
 * Recommended max resident model size: 60% of the not-visible cap, leaving headroom in the
 * same budget for the service's own heap, JNA/UniFFI overhead, and mmap bookkeeping.
 */
fun recommendedMaxModelBytes(totalMemBytes: Long): Long =
    (notVisibleCapBytes(totalMemBytes) * 0.6).toLong()
