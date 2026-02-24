package com.example.wi3dr_kmp.camera

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.example.wi3dr_kmp.streaming.StreamingController

class CameraAnalyzer(
    private val streamingController: StreamingController
) : ImageAnalysis.Analyzer {
    override fun analyze(image: ImageProxy) {
        try {
            streamingController.onFrameAvailable { _ ->
                image.toJpegBytes()
            }
        } finally {
            image.close()
        }
    }

}
