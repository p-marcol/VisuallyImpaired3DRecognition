package com.example.wi3dr_kmp.streaming

import com.example.wi3dr_kmp.network.FrameSocketClient
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.launch
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.TimeMark
import kotlin.time.TimeSource

class StreamingController {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private val _uiState = MutableStateFlow(StreamingUiState())
    val uiState: StateFlow<StreamingUiState> = _uiState.asStateFlow()
    private val _connectionErrors = MutableSharedFlow<String>(extraBufferCapacity = 1)
    val connectionErrors: SharedFlow<String> = _connectionErrors.asSharedFlow()

    private var socketClient: FrameSocketClient? = null
    private var lastSentMark: TimeMark? = null

    fun updateIp(ip: String) {
        _uiState.value = _uiState.value.copy(ip = ip)
    }

    fun updatePort(port: String) {
        _uiState.value = _uiState.value.copy(port = port)
    }

    fun updateFps(fps: Int) {
        _uiState.value = _uiState.value.copy(fps = fps.coerceIn(1, 60))
    }

    fun updateImageQualityPreset(preset: ImageQualityPreset) {
        _uiState.value = _uiState.value.copy(imageQualityPreset = preset)
    }

    fun toggleStreaming() {
        if (_uiState.value.isStreaming) {
            stopStreaming()
            return
        }

        val state = _uiState.value
        val ip = state.ip.trim()
        val port = state.port.trim().toIntOrNull() ?: return
        startStreaming(ip = ip, port = port, fps = state.fps)
    }

    fun startStreaming(ip: String, port: Int, fps: Int) {
        updateIp(ip)
        updatePort(port.toString())
        updateFps(fps)
        _uiState.value = _uiState.value.copy(
            isStreaming = true,
            connectionStatus = ConnectionStatus.Connecting
        )
        lastSentMark = null

        val previousClient = socketClient
        val newClient = FrameSocketClient("ws://$ip:$port")
        socketClient = newClient

        scope.launch {
            previousClient?.close()
            val connectResult = runCatching {
                newClient.connect()
            }

            if (socketClient !== newClient) {
                if (connectResult.isSuccess) {
                    runCatching { newClient.close() }
                }
                return@launch
            }

            connectResult
                .onSuccess {
                    _uiState.value = _uiState.value.copy(
                        isStreaming = true,
                        connectionStatus = ConnectionStatus.Connected
                    )
                }
                .onFailure { error ->
                if (socketClient === newClient) {
                    socketClient = null
                    _uiState.value = _uiState.value.copy(
                        isStreaming = false,
                        connectionStatus = ConnectionStatus.Error
                    )
                }
                val message = error.message?.takeIf { it.isNotBlank() } ?: "Unknown error"
                _connectionErrors.tryEmit("Connection failed: $message")
            }
        }
    }

    fun stopStreaming() {
        _uiState.value = _uiState.value.copy(
            isStreaming = false,
            connectionStatus = ConnectionStatus.Disconnected
        )
        lastSentMark = null

        val clientToClose = socketClient
        socketClient = null
        scope.launch {
            clientToClose?.close()
        }
    }

    fun onFrameAvailable(encodeFrameBytes: (ImageQualityPreset) -> ByteArray?) {
        val state = _uiState.value
        if (!state.isStreaming || state.connectionStatus != ConnectionStatus.Connected) return
        val client = socketClient ?: return

        val frameInterval = (1000L / state.fps.coerceAtLeast(1)).milliseconds
        val previousMark = lastSentMark
        if (previousMark != null && previousMark.elapsedNow() < frameInterval) {
            return
        }

        lastSentMark = TimeSource.Monotonic.markNow()

        val bytes = encodeFrameBytes(state.imageQualityPreset) ?: return
        scope.launch {
            runCatching {
                client.send(bytes)
            }
        }
    }

    fun dispose() {
        val clientToClose = socketClient
        socketClient = null
        _uiState.value = _uiState.value.copy(
            isStreaming = false,
            connectionStatus = ConnectionStatus.Disconnected
        )
        lastSentMark = null

        CoroutineScope(Dispatchers.Default).launch {
            clientToClose?.close()
        }
        scope.cancel()
    }

    companion object {
        const val DEFAULT_FPS = 10
    }
}
