package com.example.wi3dr_kmp.streaming

data class StreamingUiState(
    val ip: String = "192.168.1.16",
    val port: String = "8765",
    val fps: Int = StreamingController.DEFAULT_FPS,
    val imageQualityPreset: ImageQualityPreset = ImageQualityPreset.P1080,
    val isStreaming: Boolean = false,
    val connectionStatus: ConnectionStatus = ConnectionStatus.Disconnected
) {
    val isStartConfigValid: Boolean
        get() = ip.trim().isNotEmpty() && port.trim().toIntOrNull() != null

    val isLive: Boolean
        get() = isStreaming && connectionStatus == ConnectionStatus.Connected
}
