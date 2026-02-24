package com.example.wi3dr_kmp.streaming

enum class ImageQualityPreset(
    val shortLabel: String,
    val targetWidth: Int,
    val targetHeight: Int,
) {
    P720("720p", 960, 720),
    P1080("1080p", 1440, 1080),
    P2K("2K", 2048, 1536),
    P4K("4K", 2880, 2160),
    ;

    val label: String
        get() = "${targetWidth}x${targetHeight}"
}
