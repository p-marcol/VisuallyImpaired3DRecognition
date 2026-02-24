package com.example.wi3dr_kmp.input

import androidx.compose.ui.text.input.KeyboardType

expect object PlatformKeyboardHints {
    val ipKeyboardType: KeyboardType
    val portKeyboardType: KeyboardType
}
