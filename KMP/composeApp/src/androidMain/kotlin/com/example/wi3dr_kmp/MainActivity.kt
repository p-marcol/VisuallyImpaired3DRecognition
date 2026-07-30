package com.example.wi3dr_kmp

import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.safeContentPadding
import androidx.compose.foundation.layout.width
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.lifecycle.lifecycleScope
import com.example.wi3dr_kmp.camera.CameraAnalyzer
import com.example.wi3dr_kmp.camera.toAndroidTargetResolution
import com.example.wi3dr_kmp.discovery.MDNS_DISCOVERY_TIMEOUT_MS
import com.example.wi3dr_kmp.discovery.MDNS_LOG_TAG
import com.example.wi3dr_kmp.discovery.MDNS_SERVICE_NAME
import com.example.wi3dr_kmp.discovery.MDNS_SERVICE_TYPE
import com.example.wi3dr_kmp.discovery.MdnsServerScanner
import com.example.wi3dr_kmp.input.VoiceCommandRecognizer
import com.example.wi3dr_kmp.streaming.ImageQualityPreset
import com.example.wi3dr_kmp.streaming.StreamingController
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private val streamingController = StreamingController()
    private val mdnsServerScanner by lazy { MdnsServerScanner(this) }
    private val cameraAnalysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var appliedCameraQualityPreset: ImageQualityPreset? = null
    private var isScanInProgress by mutableStateOf(false)
    private var voiceCommandRecognizer: VoiceCommandRecognizer? = null
    private var isVoiceRecognitionAvailable by mutableStateOf(false)
    private var isVoiceRecognitionInProgress by mutableStateOf(false)
    private var voiceRecognitionStatus by mutableStateOf<String?>(null)
    private var textToSpeech: TextToSpeech? = null
    private var isTextToSpeechReady = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        initializeTextToSpeech()
        observeConnectionErrors()
        observeInfoResponses()

        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        WindowCompat.setDecorFitsSystemWindows(window, true)

        if (checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(
                arrayOf(android.Manifest.permission.CAMERA),
                CAMERA_PERMISSION_REQUEST_CODE
            )
            return
        }

        initializeApp()
    }

    private fun initializeApp() {
        previewView = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
        }
        initializeVoiceCommandRecognizer()

        setContent {
            val uiState by streamingController.uiState.collectAsState()
            Row(
                modifier = Modifier
                    .fillMaxSize()
                    .safeContentPadding(),
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                App(
                    streamingController = streamingController,
                    modifier = Modifier.width(300.dp),
                    onScanClick = ::scanForServerInLan,
                    isScanInProgress = isScanInProgress,
                    showVoiceRecognitionSection = true,
                    isVoiceRecognitionAvailable = isVoiceRecognitionAvailable,
                    isVoiceRecognitionInProgress = isVoiceRecognitionInProgress,
                    voiceRecognitionStatus = voiceRecognitionStatus,
                    onVoiceCommandClick = ::startVoiceCommandRecognition
                )

                AndroidApp(
                    previewView,
                    isLive = uiState.isLive,
                    modifier = Modifier.weight(1f)
                )
            }
        }

        startCamera(streamingController.uiState.value.imageQualityPreset)
        observeCameraQualityChanges()
    }

    private fun initializeVoiceCommandRecognizer() {
        val recognizer = VoiceCommandRecognizer(
            context = this,
            onInfoCommand = {
                streamingController.requestObjectInfo()
                Toast.makeText(this, "Info request sent", Toast.LENGTH_SHORT).show()
            },
            onStatusChanged = { status ->
                voiceRecognitionStatus = status
            },
            onListeningChanged = { isListening ->
                isVoiceRecognitionInProgress = isListening
            }
        )
        voiceCommandRecognizer = recognizer
        isVoiceRecognitionAvailable = recognizer.isAvailable
        if (!recognizer.isAvailable) {
            voiceRecognitionStatus = "Voice recognition unavailable"
        }
    }

    private fun startCamera(qualityPreset: ImageQualityPreset) {
        appliedCameraQualityPreset = qualityPreset
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().apply {
                surfaceProvider = previewView.surfaceProvider
            }

            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(qualityPreset.toAndroidTargetResolution())
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(
                        cameraAnalysisExecutor,
                        CameraAnalyzer(streamingController = streamingController)
                    )
                }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun observeCameraQualityChanges() {
        lifecycleScope.launch {
            streamingController.uiState.collect { state ->
                val preset = state.imageQualityPreset
                if (appliedCameraQualityPreset != preset) {
                    startCamera(preset)
                }
            }
        }
    }

    private fun observeConnectionErrors() {
        lifecycleScope.launch {
            streamingController.connectionErrors.collect { message ->
                Toast.makeText(this@MainActivity, message, Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun observeInfoResponses() {
        lifecycleScope.launch {
            streamingController.infoResponses.collect { message ->
                Toast.makeText(this@MainActivity, message, Toast.LENGTH_LONG).show()
                speakInfoResponse(message)
            }
        }
    }

    private fun initializeTextToSpeech() {
        textToSpeech = TextToSpeech(this) { status ->
            if (status != TextToSpeech.SUCCESS) {
                isTextToSpeechReady = false
                return@TextToSpeech
            }

            val tts = textToSpeech ?: return@TextToSpeech
            val languageResult = tts.setLanguage(Locale.getDefault())
            isTextToSpeechReady = languageResult != TextToSpeech.LANG_MISSING_DATA &&
                languageResult != TextToSpeech.LANG_NOT_SUPPORTED
        }
    }

    private fun speakInfoResponse(message: String) {
        if (!isTextToSpeechReady) return

        textToSpeech?.speak(
            message,
            TextToSpeech.QUEUE_FLUSH,
            null,
            INFO_RESPONSE_UTTERANCE_ID
        )
    }

    private fun startVoiceCommandRecognition() {
        if (checkSelfPermission(android.Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(
                arrayOf(android.Manifest.permission.RECORD_AUDIO),
                RECORD_AUDIO_PERMISSION_REQUEST_CODE
            )
            return
        }

        voiceCommandRecognizer?.startListening()
    }

    private fun scanForServerInLan() {
        if (isScanInProgress) {
            Log.d(MDNS_LOG_TAG, "Scan ignored: already in progress.")
            return
        }
        lifecycleScope.launch {
            isScanInProgress = true
            Log.d(
                MDNS_LOG_TAG,
                "Scan started. targetName=$MDNS_SERVICE_NAME, targetType=$MDNS_SERVICE_TYPE, timeoutMs=$MDNS_DISCOVERY_TIMEOUT_MS"
            )
            try {
                val discovered = runCatching { mdnsServerScanner.discoverServer() }.getOrNull()
                if (discovered == null) {
                    Log.d(MDNS_LOG_TAG, "Scan finished: no matching server found.")
                    Toast.makeText(
                        this@MainActivity,
                        "VI3DR Server not found on LAN.",
                        Toast.LENGTH_SHORT
                    ).show()
                    return@launch
                }

                Log.i(
                    MDNS_LOG_TAG,
                    "Scan finished: matched server ip=${discovered.ip}, port=${discovered.port}"
                )
                streamingController.updateIp(discovered.ip)
                streamingController.updatePort(discovered.port.toString())
                Toast.makeText(
                    this@MainActivity,
                    "Server found: ${discovered.ip}:${discovered.port}",
                    Toast.LENGTH_SHORT
                ).show()
            } finally {
                isScanInProgress = false
                Log.d(MDNS_LOG_TAG, "Scan ended.")
            }
        }
    }

    override fun onDestroy() {
        voiceCommandRecognizer?.dispose()
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        streamingController.dispose()
        cameraAnalysisExecutor.shutdown()
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        when (requestCode) {
            CAMERA_PERMISSION_REQUEST_CODE -> {
                if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
                    initializeApp()
                } else {
                    Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
                }
            }
            RECORD_AUDIO_PERMISSION_REQUEST_CODE -> {
                if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
                    startVoiceCommandRecognition()
                } else {
                    voiceRecognitionStatus = "Microphone permission denied"
                    Toast.makeText(this, "Microphone permission denied", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    companion object {
        private const val CAMERA_PERMISSION_REQUEST_CODE = 1001
        private const val RECORD_AUDIO_PERMISSION_REQUEST_CODE = 1002
        private const val INFO_RESPONSE_UTTERANCE_ID = "info-response"
    }
}
