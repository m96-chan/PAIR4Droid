// Root build file: declares plugin versions once (via the version catalog) so `app`
// can apply them without repeating a version number.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.compose) apply false
}
