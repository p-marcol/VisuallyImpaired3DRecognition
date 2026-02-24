package com.example.wi3dr_kmp

import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Size
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.example.wi3dr_kmp.camera.CameraAnalyzer
import com.example.wi3dr_kmp.streaming.ImageQualityPreset
import com.example.wi3dr_kmp.streaming.StreamingController
import java.util.concurrent.Executors
import androidx.camera.core.Preview
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.safeContentPadding
import androidx.compose.foundation.layout.width
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.view.WindowCompat
import androidx.lifecycle.lifecycleScope
import java.util.concurrent.ExecutorService
import kotlinx.coroutines.launch


class MainActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private val streamingController = StreamingController()
    private val cameraAnalysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var appliedCameraQualityPreset: ImageQualityPreset? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        observeConnectionErrors()

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
            val uiState by streamingController.uiState.collectAsState()
            Row(
                modifier = Modifier
                    .fillMaxSize()
                    .safeContentPadding(),
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // LEFT SIDE: UI
                App(
                    streamingController = streamingController,
                    modifier = Modifier.width(300.dp)
                )

                // RIGHT SIDE: Camera Preview
                AndroidApp(
                    previewView,
                    isLive = uiState.isLive,
                    modifier = Modifier.weight(1f)
                )
            }
        }

        startCamera(streamingController.uiState.value.imageQualityPreset)
        observeCameraQualityChanges()

    }

    private fun startCamera(qualityPreset: ImageQualityPreset) {
        appliedCameraQualityPreset = qualityPreset
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().apply {
                surfaceProvider = previewView.surfaceProvider
            }

            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(qualityPreset.toAndroidTargetResolution())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(
                        cameraAnalysisExecutor,
                        CameraAnalyzer(streamingController = streamingController)
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

    private fun observeCameraQualityChanges() {
        lifecycleScope.launch {
            streamingController.uiState.collect { state ->
                val preset = state.imageQualityPreset
                if (appliedCameraQualityPreset != preset) {
                    startCamera(preset)
                }
            }
        }
    }

    private fun observeConnectionErrors() {
        lifecycleScope.launch {
            streamingController.connectionErrors.collect { message ->
                Toast.makeText(this@MainActivity, message, Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onDestroy() {
        streamingController.dispose()
        cameraAnalysisExecutor.shutdown()
        super.onDestroy()
    }
}

private fun ImageQualityPreset.toAndroidTargetResolution(): Size = when (this) {
    else -> Size(targetWidth, targetHeight)
}
