package com.example.wi3dr_kmp

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.safeContentPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.UIKitView
import com.example.wi3dr_kmp.camera.IosCameraManager
import com.example.wi3dr_kmp.streaming.StreamingController
import kotlinx.coroutines.delay

@Composable
fun IOSApp(
    streamingController: StreamingController,
    modifier: Modifier = Modifier
) {
    val uiState by streamingController.uiState.collectAsState()
    val cameraManager = remember(streamingController) { IosCameraManager(streamingController) }
    var transientError by remember { mutableStateOf<String?>(null) }

    DisposableEffect(cameraManager) {
        cameraManager.start()
        onDispose {
            cameraManager.dispose()
            streamingController.dispose()
        }
    }

    LaunchedEffect(streamingController) {
        streamingController.connectionErrors.collect { message ->
            transientError = message
        }
    }

    LaunchedEffect(uiState.imageQualityPreset) {
        cameraManager.updateImageQualityPreset(uiState.imageQualityPreset)
    }

    LaunchedEffect(transientError) {
        if (transientError == null) return@LaunchedEffect
        val current = transientError
        delay(2200)
        if (transientError == current) {
            transientError = null
        }
    }

    Row(
        modifier = modifier
            .fillMaxSize()
            .safeContentPadding()
            .padding(12.dp),
        horizontalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        App(
            streamingController = streamingController,
            modifier = Modifier.width(300.dp)
        )

        Box(
            modifier = Modifier.weight(1f)
        ) {
            Box(
                modifier = Modifier
                    .align(Alignment.Center)
                    .fillMaxHeight()
                    .aspectRatio(4f / 3f)
            ) {
                UIKitView(
                    factory = { cameraManager.previewView() },
                    modifier = Modifier.fillMaxSize()
                )

                if (uiState.isLive) {
                    LiveBadge(
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .padding(12.dp)
                    )
                }
            }

            transientError?.let { message ->
                Text(
                    text = message,
                    color = Color.White,
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .padding(top = 12.dp)
                        .background(
                            color = Color(0xCC202124),
                            shape = RoundedCornerShape(10.dp)
                        )
                        .padding(horizontal = 12.dp, vertical = 8.dp)
                )
            }
        }
    }
}

@Composable
private fun LiveBadge(modifier: Modifier = Modifier) {
    Text(
        text = "LIVE",
        color = Color.White,
        modifier = modifier
            .background(
                color = Color(0xFFD32F2F),
                shape = RoundedCornerShape(8.dp)
            )
            .padding(horizontal = 10.dp, vertical = 6.dp)
    )
}
