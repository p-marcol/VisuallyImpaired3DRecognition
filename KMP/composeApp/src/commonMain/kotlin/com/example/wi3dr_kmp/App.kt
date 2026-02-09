package com.example.wi3dr_kmp

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier

@Composable
fun App(
    onStartStream: (ip: String, port: Int, fps: Int) -> Unit,
    onStopStream: () -> Unit,
    onFpsChanged: (fps: Int) -> Unit,
    modifier: Modifier = Modifier
) {
    MaterialTheme {
        var ip by remember { mutableStateOf("192.168.1.16") }
        var port by remember { mutableStateOf("8765") }
        var fps by remember { mutableStateOf(10f) }
        var streaming by remember { mutableStateOf(false) }

        Column(
            modifier = modifier
        ) {
            OutlinedTextField(
                value = ip,
                onValueChange = { ip = it },
                label = { Text("IP Address") })
            OutlinedTextField(value = port, onValueChange = { port = it }, label = { Text("Port") })
            Text("FPS: ${fps.toInt()}")
            Slider(value = fps, onValueChange = {
                fps = it
                onFpsChanged(it.toInt())
            }, valueRange = 1f..60f)

            Button(onClick = {
                streaming = !streaming
                if (streaming) onStartStream(
                    ip,
                    port.trim().toInt(),
                    fps.toInt()
                ) else onStopStream()
            }, modifier = Modifier.fillMaxWidth()) {
                Text(if (streaming) "Stop Streaming" else "Start Streaming")
            }
        }
    }
}