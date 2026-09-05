package com.pair4droid.util

import android.os.PowerManager
import uniffi.pair_ffi.ThermalStatus

/**
 * Maps [PowerManager]'s `THERMAL_STATUS_*` constants (API 29+) onto the FFI's thermal enum.
 *
 * ASSUMPTION (flagged for the design session): pair-ffi/src/lib.rs's doc comment names a
 * `thermal` field on `ExternalSignals` but does not name its Rust enum or list variants. This
 * assumes a UniFFI enum `ThermalStatus` with one variant per Android thermal level
 * (NONE/LIGHT/MODERATE/SEVERE/CRITICAL/EMERGENCY/SHUTDOWN) — confirm against the real
 * pair-ffi type once implemented.
 */
fun androidThermalStatusToFfi(status: Int): ThermalStatus = when (status) {
    PowerManager.THERMAL_STATUS_NONE -> ThermalStatus.NONE
    PowerManager.THERMAL_STATUS_LIGHT -> ThermalStatus.LIGHT
    PowerManager.THERMAL_STATUS_MODERATE -> ThermalStatus.MODERATE
    PowerManager.THERMAL_STATUS_SEVERE -> ThermalStatus.SEVERE
    PowerManager.THERMAL_STATUS_CRITICAL -> ThermalStatus.CRITICAL
    PowerManager.THERMAL_STATUS_EMERGENCY -> ThermalStatus.EMERGENCY
    PowerManager.THERMAL_STATUS_SHUTDOWN -> ThermalStatus.SHUTDOWN
    else -> ThermalStatus.NONE
}
