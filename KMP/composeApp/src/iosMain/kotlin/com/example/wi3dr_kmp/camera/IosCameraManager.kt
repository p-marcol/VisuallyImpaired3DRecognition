package com.example.wi3dr_kmp.camera

import com.example.wi3dr_kmp.streaming.ImageQualityPreset
import com.example.wi3dr_kmp.streaming.StreamingController
import kotlin.concurrent.Volatile
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.ObjCObjectVar
import kotlinx.cinterop.alloc
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.readValue
import kotlinx.cinterop.usePinned
import kotlinx.cinterop.value
import platform.AVFoundation.AVCaptureConnection
import platform.AVFoundation.AVCaptureDevice
import platform.AVFoundation.AVCaptureDeviceInput
import platform.AVFoundation.AVCaptureOutput
import platform.AVFoundation.AVCaptureSession
import platform.AVFoundation.AVCaptureSessionPreset1280x720
import platform.AVFoundation.AVCaptureSessionPreset1920x1080
import platform.AVFoundation.AVCaptureSessionPreset3840x2160
import platform.AVFoundation.AVCaptureSessionPreset640x480
import platform.AVFoundation.AVCaptureSessionPresetPhoto
import platform.AVFoundation.AVCaptureVideoDataOutput
import platform.AVFoundation.AVCaptureVideoDataOutputSampleBufferDelegateProtocol
import platform.AVFoundation.AVCaptureVideoPreviewLayer
import platform.AVFoundation.AVCaptureVideoOrientationLandscapeRight
import platform.AVFoundation.AVLayerVideoGravityResizeAspect
import platform.AVFoundation.AVMediaTypeVideo
import platform.CoreGraphics.CGImageRefVar
import platform.CoreGraphics.CGImageRelease
import platform.CoreGraphics.CGRectZero
import platform.CoreMedia.CMSampleBufferGetImageBuffer
import platform.CoreMedia.CMSampleBufferRef
import platform.CoreVideo.CVPixelBufferRef
import platform.CoreVideo.CVPixelBufferRelease
import platform.CoreVideo.CVPixelBufferRetain
import platform.CoreVideo.kCVPixelBufferPixelFormatTypeKey
import platform.CoreVideo.kCVPixelFormatType_32BGRA
import platform.Foundation.NSData
import platform.Foundation.NSError
import platform.UIKit.UIImage
import platform.UIKit.UIImageJPEGRepresentation
import platform.UIKit.UIView
import platform.VideoToolbox.VTCreateCGImageFromCVPixelBuffer
import platform.darwin.NSObject
import platform.darwin.dispatch_async
import platform.darwin.dispatch_queue_create
import platform.posix.memcpy

@OptIn(ExperimentalForeignApi::class)
class IosCameraManager(
    private val streamingController: StreamingController
) : NSObject(), AVCaptureVideoDataOutputSampleBufferDelegateProtocol {

    private val sessionQueue = dispatch_queue_create("wi3dr.camera.session", null)
    private val outputQueue = dispatch_queue_create("wi3dr.camera.output", null)
    private val encodeQueue = dispatch_queue_create("wi3dr.camera.encode", null)
    private val captureSession = AVCaptureSession()
    private val videoOutput = AVCaptureVideoDataOutput()

    private var previewLayer: AVCaptureVideoPreviewLayer? = null
    private var previewContainerView: PreviewContainerView? = null
    private var isConfigured = false
    @Volatile
    private var selectedQualityPreset = ImageQualityPreset.P1080
    @Volatile
    private var encodeInFlight = false

    fun previewView(): UIView {
        val existing = previewContainerView
        if (existing != null) return existing

        val container = PreviewContainerView()
        val layer = AVCaptureVideoPreviewLayer(session = captureSession)
        layer.videoGravity = AVLayerVideoGravityResizeAspect
        container.attachPreviewLayer(layer)

        previewLayer = layer
        previewContainerView = container
        applyLandscapeOrientation()
        return container
    }

    fun start() {
        startSession()
    }

    fun stop() {
        dispatch_async(sessionQueue) {
            if (captureSession.running) {
                captureSession.stopRunning()
            }
        }
    }

    fun updateImageQualityPreset(preset: ImageQualityPreset) {
        selectedQualityPreset = preset
        dispatch_async(sessionQueue) {
            if (isConfigured) {
                captureSession.beginConfiguration()
            }
            applySessionPreset(preset)
            if (isConfigured) {
                captureSession.commitConfiguration()
            }
        }
    }

    fun dispose() {
        stop()
        videoOutput.setSampleBufferDelegate(null, null)
    }

    private fun startSession() {
        dispatch_async(sessionQueue) {
            configureSessionIfNeeded()
            if (!captureSession.running) {
                captureSession.startRunning()
            }
        }
    }

    private fun configureSessionIfNeeded() {
        if (isConfigured) return

        val device = AVCaptureDevice.defaultDeviceWithMediaType(AVMediaTypeVideo) ?: return

        memScoped {
            val errorPtr = alloc<ObjCObjectVar<NSError?>>()
            val input = AVCaptureDeviceInput.deviceInputWithDevice(device, errorPtr.ptr)
                ?: return

            captureSession.beginConfiguration()
            applySessionPreset(selectedQualityPreset)

            if (captureSession.canAddInput(input)) {
                captureSession.addInput(input)
            }

            videoOutput.videoSettings = mapOf(
                kCVPixelBufferPixelFormatTypeKey to kCVPixelFormatType_32BGRA
            )
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.setSampleBufferDelegate(this@IosCameraManager, outputQueue)

            if (captureSession.canAddOutput(videoOutput)) {
                captureSession.addOutput(videoOutput)
            }

            applyLandscapeOrientation()
            captureSession.commitConfiguration()
            isConfigured = true
        }
    }

    private fun applyLandscapeOrientation() {
        previewLayer?.connection?.let { connection ->
            if (connection.supportsVideoOrientation) {
                connection.videoOrientation = AVCaptureVideoOrientationLandscapeRight
            }
        }

        videoOutput.connectionWithMediaType(AVMediaTypeVideo)?.let { connection ->
            if (connection.supportsVideoOrientation) {
                connection.videoOrientation = AVCaptureVideoOrientationLandscapeRight
            }
        }
    }

    override fun captureOutput(
        output: AVCaptureOutput,
        didOutputSampleBuffer: CMSampleBufferRef?,
        fromConnection: AVCaptureConnection
    ) {
        val sampleBuffer = didOutputSampleBuffer ?: return
        val pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) ?: return

        if (encodeInFlight) return
        encodeInFlight = true
        CVPixelBufferRetain(pixelBuffer)

        dispatch_async(encodeQueue) {
            try {
                streamingController.onFrameAvailable { _ ->
                    pixelBuffer.toJpegBytes()
                }
            } finally {
                CVPixelBufferRelease(pixelBuffer)
                encodeInFlight = false
            }
        }
    }

    private fun applySessionPreset(preset: ImageQualityPreset) {
        val requestedPreset = preset.toIosSessionPreset()
        captureSession.sessionPreset = when {
            requestedPreset != null && captureSession.canSetSessionPreset(requestedPreset) -> requestedPreset
            captureSession.canSetSessionPreset(AVCaptureSessionPresetPhoto) -> AVCaptureSessionPresetPhoto
            captureSession.canSetSessionPreset(AVCaptureSessionPreset640x480) -> AVCaptureSessionPreset640x480
            captureSession.canSetSessionPreset(AVCaptureSessionPreset1920x1080) -> AVCaptureSessionPreset1920x1080
            captureSession.canSetSessionPreset(AVCaptureSessionPreset1280x720) -> AVCaptureSessionPreset1280x720
            else -> captureSession.sessionPreset
        }
    }
}

@OptIn(ExperimentalForeignApi::class)
private class PreviewContainerView : UIView(frame = CGRectZero.readValue()) {
    private var attachedPreviewLayer: AVCaptureVideoPreviewLayer? = null

    fun attachPreviewLayer(layer: AVCaptureVideoPreviewLayer) {
        attachedPreviewLayer = layer
        this.layer.addSublayer(layer)
        layer.frame = bounds
    }

    override fun layoutSubviews() {
        super.layoutSubviews()
        attachedPreviewLayer?.frame = bounds
    }
}

@OptIn(ExperimentalForeignApi::class)
private fun CVPixelBufferRef.toJpegBytes(): ByteArray? {
    val pixelBuffer = this
    return memScoped {
        val cgImageRef = alloc<CGImageRefVar>()
        val status = VTCreateCGImageFromCVPixelBuffer(pixelBuffer, null, cgImageRef.ptr)
        if (status != 0) return@memScoped null

        val cgImage = cgImageRef.value ?: return@memScoped null
        try {
            val image = UIImage.imageWithCGImage(cgImage)
            val jpegData = UIImageJPEGRepresentation(image, 0.85) ?: return@memScoped null
            jpegData.toByteArray()
        } finally {
            CGImageRelease(cgImage)
        }
    }
}

private fun ImageQualityPreset.toIosSessionPreset(): String? = when (this) {
    ImageQualityPreset.P720 -> AVCaptureSessionPresetPhoto
    ImageQualityPreset.P1080 -> AVCaptureSessionPresetPhoto
    ImageQualityPreset.P2K -> AVCaptureSessionPresetPhoto
    ImageQualityPreset.P4K -> AVCaptureSessionPresetPhoto
}

@OptIn(ExperimentalForeignApi::class)
private fun NSData.toByteArray(): ByteArray {
    val size = length.toInt()
    if (size <= 0) return ByteArray(0)

    return ByteArray(size).also { bytesArray ->
        bytesArray.usePinned { pinned ->
            memcpy(pinned.addressOf(0), bytes, length)
        }
    }
}
