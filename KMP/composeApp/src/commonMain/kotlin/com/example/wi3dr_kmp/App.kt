package com.example.wi3dr_kmp

import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import com.example.wi3dr_kmp.input.PlatformKeyboardHints
import com.example.wi3dr_kmp.streaming.ImageQualityPreset
import com.example.wi3dr_kmp.streaming.StreamingController

@Composable
fun App(
    streamingController: StreamingController,
    modifier: Modifier = Modifier
) {
    MaterialTheme {
        val uiState by streamingController.uiState.collectAsState()
        val focusManager = LocalFocusManager.current
        val keyboardController = LocalSoftwareKeyboardController.current
        val portFocusRequester = remember { FocusRequester() }
        var qualityMenuExpanded by remember { mutableStateOf(false) }

        Column(modifier = modifier.verticalScroll(rememberScrollState())) {
            OutlinedTextField(
                value = uiState.ip,
                onValueChange = {
                    val sanitized = it
                        .replace("\n", "")
                        .filter { ch -> ch.isDigit() || ch == '.' }
                    streamingController.updateIp(sanitized)
                },
                label = { Text("IP Address") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(
                    keyboardType = PlatformKeyboardHints.ipKeyboardType,
                    imeAction = ImeAction.Next
                ),
                keyboardActions = KeyboardActions(
                    onNext = { portFocusRequester.requestFocus() },
                    onDone = {
                        keyboardController?.hide()
                        focusManager.clearFocus()
                    }
                ),
                modifier = Modifier.fillMaxWidth()
            )
            OutlinedTextField(
                value = uiState.port,
                onValueChange = {
                    val sanitized = it
                        .replace("\n", "")
                        .filter { ch -> ch.isDigit() }
                    streamingController.updatePort(sanitized)
                },
                label = { Text("Port") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(
                    keyboardType = PlatformKeyboardHints.portKeyboardType,
                    imeAction = ImeAction.Done
                ),
                keyboardActions = KeyboardActions(
                    onDone = {
                        keyboardController?.hide()
                        focusManager.clearFocus()
                    }
                ),
                modifier = Modifier
                    .fillMaxWidth()
                    .focusRequester(portFocusRequester)
            )
            Text("FPS: ${uiState.fps}")
            Slider(
                value = uiState.fps.toFloat(),
                onValueChange = { streamingController.updateFps(it.toInt()) },
                valueRange = 1f..60f
            )
            Text("Quality: ${uiState.imageQualityPreset.label}")
            Column {
                OutlinedButton(
                    onClick = { qualityMenuExpanded = true },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(uiState.imageQualityPreset.label)
                }
                DropdownMenu(
                    expanded = qualityMenuExpanded,
                    onDismissRequest = { qualityMenuExpanded = false }
                ) {
                    ImageQualityPreset.entries.forEach { preset ->
                        DropdownMenuItem(
                            text = { Text(preset.label) },
                            onClick = {
                                streamingController.updateImageQualityPreset(preset)
                                qualityMenuExpanded = false
                            }
                        )
                    }
                }
            }
            Text("Status: ${uiState.connectionStatus.label}")

            Button(
                onClick = streamingController::toggleStreaming,
                enabled = uiState.isStreaming || uiState.isStartConfigValid,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(if (uiState.isStreaming) "Stop Streaming" else "Start Streaming")
            }
        }
    }
}
