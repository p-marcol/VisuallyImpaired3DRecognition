package com.example.wi3dr_kmp.network

import io.ktor.client.HttpClient
import io.ktor.client.plugins.websocket.WebSockets
import io.ktor.client.plugins.websocket.webSocketSession
import io.ktor.websocket.Frame
import io.ktor.websocket.WebSocketSession
import io.ktor.websocket.close
import io.ktor.websocket.readText

class FrameSocketClient(
    private val url: String
) {
    private var client = HttpClient {
        install(WebSockets)
    }

    private var session: WebSocketSession? = null

    suspend fun connect() {
        session = client.webSocketSession(url)
    }

    suspend fun send(bytes: ByteArray) {
        session?.send(Frame.Binary(true, bytes))
    }

    suspend fun sendStop() {
        session?.send(Frame.Text("stop"))
    }

    suspend fun awaitNextTextMessage(): String? {
        val activeSession = session ?: return null

        while (true) {
            when (val frame = activeSession.incoming.receive()) {
                is Frame.Text -> return frame.readText()
                is Frame.Close -> return null
                else -> Unit
            }
        }
    }

    suspend fun close() {
        session?.close()
        session = null
        client.close()
    }

}
