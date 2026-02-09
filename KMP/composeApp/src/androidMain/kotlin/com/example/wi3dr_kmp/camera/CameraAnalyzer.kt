package com.example.wi3dr_kmp.camera

import android.graphics.Bitmap
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.wi3dr_kmp.network.FrameSocketClient
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream

class CameraAnalyzer(
    private val getSocketClient: () -> FrameSocketClient?,
    private val isStreaming: () -> Boolean,
    private val getFps: () -> Int
) : ImageAnalysis.Analyzer {
    private val scope = CoroutineScope(Dispatchers.IO)
    private var lastSent = 0L

    override fun analyze(image: ImageProxy) {

        if (!isStreaming()) {
            image.close()
            return
        }


        val fps = getFps().coerceAtLeast(1)
        val frameInternalMs = 1000L / fps

        val now = System.currentTimeMillis()
        if (now - lastSent < frameInternalMs) {
            image.close()
            return
        }
        lastSent = now

        val bitmap = image.toBitmap()
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 80, stream)

        val bytes = stream.toByteArray()

        val socketClient = getSocketClient() ?: run {
            image.close()
            return
        }

        scope.launch {
            socketClient.send(bytes)
        }

        image.close()
    }

}