package com.example.wi3dr_kmp

import androidx.compose.runtime.remember
import androidx.compose.ui.window.ComposeUIViewController
import com.example.wi3dr_kmp.streaming.StreamingController

fun MainViewController() = ComposeUIViewController {
    val streamingController = remember { StreamingController() }
    IOSApp(streamingController = streamingController)
}
