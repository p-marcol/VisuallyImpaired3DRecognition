package com.example.wi3dr_kmp.input

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import java.util.Locale

private const val RESTART_DELAY_MS = 350L
private const val BUSY_RESTART_DELAY_MS = 1000L

class VoiceCommandRecognizer(
    context: Context,
    private val onInfoCommand: () -> Unit,
    private val onStatusChanged: (String?) -> Unit,
    private val onListeningChanged: (Boolean) -> Unit
) {
    private val appContext = context.applicationContext
    private val mainHandler = Handler(Looper.getMainLooper())
    private val speechRecognizer = if (SpeechRecognizer.isRecognitionAvailable(appContext)) {
        SpeechRecognizer.createSpeechRecognizer(appContext)
    } else {
        null
    }
    private var isContinuousListeningEnabled = false
    private var isRecognitionActive = false

    val isAvailable: Boolean
        get() = speechRecognizer != null

    private val recognizerIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(
            RecognizerIntent.EXTRA_LANGUAGE_MODEL,
            RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
        )
        putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.US.toLanguageTag())
        putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, false)
        putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 5)
    }

    init {
        speechRecognizer?.setRecognitionListener(
            object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {
                    onListeningChanged(true)
                    onStatusChanged("Listening")
                }

                override fun onBeginningOfSpeech() = Unit

                override fun onRmsChanged(rmsdB: Float) = Unit

                override fun onBufferReceived(buffer: ByteArray?) = Unit

                override fun onEndOfSpeech() {
                    onListeningChanged(false)
                    onStatusChanged("Processing voice command")
                }

                override fun onError(error: Int) {
                    isRecognitionActive = false
                    onListeningChanged(false)

                    if (!isContinuousListeningEnabled) return

                    onStatusChanged(error.toStatusMessage())
                    scheduleRestart(error.restartDelayMs())
                }

                override fun onResults(results: Bundle?) {
                    isRecognitionActive = false
                    onListeningChanged(false)
                    val matches = results
                        ?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        .orEmpty()

                    if (matches.any { it.hasInfoCommand() }) {
                        onStatusChanged("Voice command: info")
                        onInfoCommand()
                    } else {
                        onStatusChanged("Voice command not recognized")
                    }
                    scheduleRestart()
                }

                override fun onPartialResults(partialResults: Bundle?) = Unit

                override fun onEvent(eventType: Int, params: Bundle?) = Unit
            }
        )
    }

    fun startContinuousListening() {
        isContinuousListeningEnabled = true
        startListeningSession()
    }

    fun stopContinuousListening() {
        isContinuousListeningEnabled = false
        mainHandler.removeCallbacksAndMessages(null)
        isRecognitionActive = false
        onListeningChanged(false)
        speechRecognizer?.cancel()
        onStatusChanged(null)
    }

    private fun startListeningSession() {
        val recognizer = speechRecognizer
        if (recognizer == null) {
            onStatusChanged("Voice recognition unavailable")
            return
        }
        if (isRecognitionActive) return

        isRecognitionActive = true
        onListeningChanged(true)
        onStatusChanged("Listening")
        recognizer.startListening(recognizerIntent)
    }

    private fun scheduleRestart(delayMs: Long = RESTART_DELAY_MS) {
        if (!isContinuousListeningEnabled) return

        mainHandler.removeCallbacksAndMessages(null)
        mainHandler.postDelayed(
            {
                if (isContinuousListeningEnabled) {
                    startListeningSession()
                }
            },
            delayMs
        )
    }

    fun dispose() {
        stopContinuousListening()
        speechRecognizer?.destroy()
    }
}

private fun String.hasInfoCommand(): Boolean {
    return lowercase(Locale.ROOT)
        .split(' ', '\n', '\t', ',', '.', '!', '?', ';', ':')
        .any { it == "info" }
}

private fun Int.toStatusMessage(): String = when (this) {
    SpeechRecognizer.ERROR_AUDIO -> "Audio error"
    SpeechRecognizer.ERROR_CLIENT -> "Voice recognition error"
    SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Microphone permission missing"
    SpeechRecognizer.ERROR_NETWORK -> "Network error"
    SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Network timeout"
    SpeechRecognizer.ERROR_NO_MATCH -> "Voice command not recognized"
    SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Voice recognition busy"
    SpeechRecognizer.ERROR_SERVER -> "Voice recognition server error"
    SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "No speech detected"
    else -> "Voice recognition error"
}

private fun Int.restartDelayMs(): Long = when (this) {
    SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> BUSY_RESTART_DELAY_MS
    else -> RESTART_DELAY_MS
}
