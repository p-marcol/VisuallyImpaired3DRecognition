package com.example.wi3dr_kmp.camera

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

fun ImageProxy.toJpegBytes(quality: Int = 85): ByteArray {
    val crop = cropRect
    val outWidth = crop.width()
    val outHeight = crop.height()
    val nv21 = yuv420888ToNv21(crop)

    val yuvImage = YuvImage(
        nv21,
        ImageFormat.NV21,
        outWidth,
        outHeight,
        null
    )

    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, outWidth, outHeight), quality, out)
    return out.toByteArray()
}

private fun ImageProxy.yuv420888ToNv21(crop: Rect): ByteArray {
    val width = crop.width()
    val height = crop.height()
    val output = ByteArray(width * height * ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8)
    val rowData = ByteArray(planes.maxOf { it.rowStride })

    var channelOffset: Int
    var outputStride: Int

    for (planeIndex in planes.indices) {
        when (planeIndex) {
            0 -> {
                channelOffset = 0
                outputStride = 1
            }
            1 -> {
                channelOffset = width * height + 1
                outputStride = 2
            }
            else -> {
                channelOffset = width * height
                outputStride = 2
            }
        }

        val plane = planes[planeIndex]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val shift = if (planeIndex == 0) 0 else 1
        val planeWidth = width shr shift
        val planeHeight = height shr shift
        val cropTop = crop.top shr shift
        val cropLeft = crop.left shr shift

        buffer.position(rowStride * cropTop + pixelStride * cropLeft)

        for (row in 0 until planeHeight) {
            val length = if (pixelStride == 1 && outputStride == 1) {
                planeWidth
            } else {
                (planeWidth - 1) * pixelStride + 1
            }

            if (pixelStride == 1 && outputStride == 1) {
                buffer.get(output, channelOffset, length)
                channelOffset += length
            } else {
                buffer.get(rowData, 0, length)
                for (col in 0 until planeWidth) {
                    output[channelOffset] = rowData[col * pixelStride]
                    channelOffset += outputStride
                }
            }

            if (row < planeHeight - 1) {
                buffer.position(buffer.position() + rowStride - length)
            }
        }
    }

    return output
}
