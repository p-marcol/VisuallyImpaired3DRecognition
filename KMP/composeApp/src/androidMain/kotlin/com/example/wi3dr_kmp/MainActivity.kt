package com.example.wi3dr_kmp

import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.wi3dr_kmp.camera.CameraAnalyzer
import com.example.wi3dr_kmp.network.FrameSocketClient
import kotlinx.coroutines.launch
import java.util.concurrent.Executors
import androidx.camera.core.Preview
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.safeContentPadding
import androidx.compose.foundation.layout.width
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.view.WindowCompat


class MainActivity : ComponentActivity() {

    private var streamingEnabled = false
    lateinit var previewView: PreviewView
    private var socketClient: FrameSocketClient? = null
    private var currentFps: Int = 10


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        WindowCompat.setDecorFitsSystemWindows(window, true)

        if (checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(
                arrayOf(android.Manifest.permission.CAMERA),
                1001
            )
            return
        }

        previewView = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
        }

        setContent {
            Row(
                modifier = Modifier
                    .fillMaxSize()
                    .safeContentPadding(),
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // LEFT SIDE: UI
                App(
                    onStartStream = { ip, port, fps -> startStreaming(ip, port, fps) },
                    onStopStream = { stopStreaming() },
                    onFpsChanged = {fps -> updateFps(fps)},
                    modifier = Modifier.width(300.dp)

                )

                // RIGHT SIDE: Camera Preview
                AndroidApp(
                    previewView,
                    modifier = Modifier.weight(1f)
                )
            }
        }

        startCamera()

    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().apply {
                surfaceProvider = previewView.surfaceProvider
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(
                        Executors.newSingleThreadExecutor(),
                        CameraAnalyzer(
                            getSocketClient = { socketClient },
                            isStreaming = { streamingEnabled },
                            getFps = { currentFps }
                        )
                    )
                }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )

        }, ContextCompat.getMainExecutor(this))
    }


    fun startStreaming(ip: String, port: Int, fps: Int) {
        streamingEnabled = true
        currentFps = fps

        socketClient = FrameSocketClient("ws://$ip:$port")
        Log.d("Stream", "Connecting to ws://$ip:$port")

        lifecycleScope.launch {
            socketClient?.connect()
        }
    }

    fun stopStreaming() {
        streamingEnabled = false
    }

    fun updateFps(fps: Int) {
        currentFps = fps
    }
}