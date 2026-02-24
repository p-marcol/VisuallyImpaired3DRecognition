package com.example.wi3dr_kmp.streaming

enum class ConnectionStatus(val label: String) {
    Disconnected("Disconnected"),
    Connecting("Connecting..."),
    Connected("Connected"),
    Error("Connection failed")
}
