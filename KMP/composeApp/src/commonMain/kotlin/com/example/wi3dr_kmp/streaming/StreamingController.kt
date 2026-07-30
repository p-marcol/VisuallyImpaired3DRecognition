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

private const val INFO_RESPONSE_KEY = "info-response"
private const val INFO_ERROR_KEY = "info-error"

class StreamingController {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private val _uiState = MutableStateFlow(StreamingUiState())
    val uiState: StateFlow<StreamingUiState> = _uiState.asStateFlow()
    private val _connectionErrors = MutableSharedFlow<String>(extraBufferCapacity = 1)
    val connectionErrors: SharedFlow<String> = _connectionErrors.asSharedFlow()
    private val _infoResponses = MutableSharedFlow<String>(extraBufferCapacity = 1)
    val infoResponses: SharedFlow<String> = _infoResponses.asSharedFlow()

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
                    observeServerCommands(newClient)
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
        stopStreaming(sendStopMessage = true)
    }

    fun requestObjectInfo() {
        val state = _uiState.value
        if (!state.isStreaming || state.connectionStatus != ConnectionStatus.Connected) return
        val client = socketClient ?: return

        scope.launch {
            runCatching {
                client.sendObjectInfoRequest()
            }
        }
    }

    private fun stopStreaming(sendStopMessage: Boolean) {
        _uiState.value = _uiState.value.copy(
            isStreaming = false,
            connectionStatus = ConnectionStatus.Disconnected
        )
        lastSentMark = null

        val clientToClose = socketClient
        socketClient = null
        scope.launch {
            if (sendStopMessage) {
                runCatching {
                    clientToClose?.sendStop()
                }
            }
            clientToClose?.close()
        }
    }

    private fun observeServerCommands(client: FrameSocketClient) {
        scope.launch {
            while (socketClient === client) {
                val message = runCatching {
                    client.awaitNextTextMessage()
                }.getOrNull()

                if (message == null) {
                    if (socketClient === client) {
                        stopStreaming(sendStopMessage = false)
                        _connectionErrors.tryEmit("Connection closed by server")
                    }
                    break
                }

                if (message.trim() == "client_stop" && socketClient === client) {
                    _connectionErrors.tryEmit("Connection closed by server")
                    stopStreaming(sendStopMessage = false)
                    break
                }

                message.toInfoResponseDisplayText()?.let { response ->
                    _infoResponses.tryEmit(response)
                }
            }
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

private fun String.toInfoResponseDisplayText(): String? {
    val message = trim()
    if (message.startsWith("$INFO_RESPONSE_KEY:")) {
        return message.substringAfter(':').trim().ifBlank { INFO_RESPONSE_KEY }
    }
    if (message.startsWith("$INFO_ERROR_KEY:")) {
        return message.substringAfter(':').trim().ifBlank { INFO_ERROR_KEY }
    }

    return extractControlValue(INFO_RESPONSE_KEY)
        ?: extractControlValue(INFO_ERROR_KEY)?.let { "Error: $it" }
}

private fun String.extractControlValue(key: String): String? {
    val quotedKey = "\"$key\""
    val keyIndex = indexOf(quotedKey)
    if (keyIndex == -1) return null

    val colonIndex = indexOf(':', startIndex = keyIndex + quotedKey.length)
    if (colonIndex == -1) return key

    val valueStart = nextNonWhitespaceIndex(colonIndex + 1) ?: return key
    return extractJsonValue(valueStart)
        ?.ifBlank { key }
        ?: key
}

private fun String.nextNonWhitespaceIndex(startIndex: Int): Int? {
    for (index in startIndex until length) {
        if (!this[index].isWhitespace()) return index
    }
    return null
}

private fun String.extractJsonValue(startIndex: Int): String? {
    return when (this[startIndex]) {
        '"' -> extractJsonString(startIndex)
        '{' -> extractBalancedJson(startIndex, '{', '}')
        '[' -> extractBalancedJson(startIndex, '[', ']')
        else -> {
            val endIndex = indexOf(',', startIndex).takeIf { it != -1 }
                ?: indexOf('}', startIndex).takeIf { it != -1 }
                ?: length
            substring(startIndex, endIndex).trim()
        }
    }
}

private fun String.extractJsonString(startIndex: Int): String? {
    val result = StringBuilder()
    var escaped = false

    for (index in startIndex + 1 until length) {
        val char = this[index]
        if (escaped) {
            result.append(
                when (char) {
                    '"' -> '"'
                    '\\' -> '\\'
                    '/' -> '/'
                    'b' -> '\b'
                    'n' -> '\n'
                    'r' -> '\r'
                    't' -> '\t'
                    else -> char
                }
            )
            escaped = false
            continue
        }

        when (char) {
            '\\' -> escaped = true
            '"' -> return result.toString()
            else -> result.append(char)
        }
    }

    return null
}

private fun String.extractBalancedJson(
    startIndex: Int,
    openingChar: Char,
    closingChar: Char
): String? {
    var depth = 0
    var inString = false
    var escaped = false

    for (index in startIndex until length) {
        val char = this[index]
        if (inString) {
            if (escaped) {
                escaped = false
            } else if (char == '\\') {
                escaped = true
            } else if (char == '"') {
                inString = false
            }
            continue
        }

        when (char) {
            '"' -> inString = true
            openingChar -> depth += 1
            closingChar -> {
                depth -= 1
                if (depth == 0) {
                    return substring(startIndex, index + 1)
                }
            }
        }
    }

    return null
}
