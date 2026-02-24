package com.example.wi3dr_kmp

import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.example.wi3dr_kmp.ui.CameraPreview

@Composable
fun AndroidApp(
    previewView: PreviewView,
    isLive: Boolean,
    modifier: Modifier = Modifier
) {
    Box(modifier = modifier) {
        CameraPreview(
            previewView = previewView,
            modifier = Modifier.fillMaxSize()
        )

        if (isLive) {
            Text(
                text = "LIVE",
                color = Color.White,
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(12.dp)
                    .background(
                        color = Color(0xFFD32F2F),
                        shape = RoundedCornerShape(8.dp)
                    )
                    .padding(horizontal = 10.dp, vertical = 6.dp)
            )
        }
    }
}
