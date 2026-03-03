package com.example.wi3dr_kmp.camera

import android.util.Size
import com.example.wi3dr_kmp.streaming.ImageQualityPreset

fun ImageQualityPreset.toAndroidTargetResolution(): Size = when (this) {
    else -> Size(targetWidth, targetHeight)
}
