package com.pair4droid

import android.app.Application
import com.pair4droid.repo.NodeRepository

class Pair4DroidApp : Application() {
    override fun onCreate() {
        super.onCreate()
        NodeRepository.init(this)
    }
}
