package com.example.wi3dr_kmp

import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.example.wi3dr_kmp.ui.CameraPreview

@Composable
fun AndroidApp(
    previewView: PreviewView,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        CameraPreview(
            previewView = previewView,
            modifier = Modifier.fillMaxSize()
        )
    }
}