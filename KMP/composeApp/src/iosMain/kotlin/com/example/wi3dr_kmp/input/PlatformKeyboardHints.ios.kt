package com.example.wi3dr_kmp.input

import androidx.compose.ui.text.input.KeyboardType

actual object PlatformKeyboardHints {
    // iOS numeric/decimal keyboards often don't expose Return/Done.
    // Use a keyboard with Enter and enforce allowed characters in onValueChange.
    actual val ipKeyboardType: KeyboardType = KeyboardType.Ascii
    actual val portKeyboardType: KeyboardType = KeyboardType.Ascii
}
