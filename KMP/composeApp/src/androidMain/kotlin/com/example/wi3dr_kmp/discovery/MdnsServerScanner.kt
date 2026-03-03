package com.example.wi3dr_kmp.discovery

import android.content.Context
import android.net.nsd.NsdManager
import android.net.nsd.NsdServiceInfo
import android.util.Log
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.coroutines.resume

class MdnsServerScanner(
    private val context: Context
) {
    suspend fun discoverServer(
        timeoutMillis: Long = MDNS_DISCOVERY_TIMEOUT_MS
    ): DiscoveredLanServer? {
        val startedAt = System.currentTimeMillis()
        for (serviceType in MDNS_DISCOVERY_SERVICE_TYPES) {
            val elapsed = System.currentTimeMillis() - startedAt
            val remaining = timeoutMillis - elapsed
            if (remaining <= 0L) {
                Log.w(
                    MDNS_LOG_TAG,
                    "Global timeout reached before trying serviceType=$serviceType."
                )
                return null
            }

            Log.d(
                MDNS_LOG_TAG,
                "Starting NSD discovery for serviceType=$serviceType (remainingMs=$remaining)"
            )

            val discovered = withTimeoutOrNull(remaining) {
                suspendCancellableCoroutine { continuation ->
                    val nsdManager = context.getSystemService(NsdManager::class.java)
                    if (nsdManager == null) {
                        Log.e(MDNS_LOG_TAG, "NsdManager is null; cannot start discovery.")
                        continuation.resume(null)
                        return@suspendCancellableCoroutine
                    }

                    val finished = AtomicBoolean(false)
                    lateinit var discoveryListener: NsdManager.DiscoveryListener

                    fun finish(result: DiscoveredLanServer?) {
                        if (!finished.compareAndSet(false, true)) return
                        Log.d(
                            MDNS_LOG_TAG,
                            "Finishing discovery for serviceType=$serviceType, result=$result"
                        )
                        runCatching { nsdManager.stopServiceDiscovery(discoveryListener) }
                        if (continuation.isActive) {
                            continuation.resume(result)
                        }
                    }

                    discoveryListener = object : NsdManager.DiscoveryListener {
                        override fun onDiscoveryStarted(serviceType: String) {
                            Log.d(MDNS_LOG_TAG, "onDiscoveryStarted: serviceType=$serviceType")
                        }

                        override fun onServiceFound(serviceInfo: NsdServiceInfo) {
                            Log.d(
                                MDNS_LOG_TAG,
                                "onServiceFound: ${serviceInfo.toLogSummary()}"
                            )
                            if (finished.get()) return
                            if (!serviceInfo.matchesVi3drServer()) {
                                Log.d(
                                    MDNS_LOG_TAG,
                                    "Ignoring service (name/type mismatch): ${serviceInfo.toLogSummary()}"
                                )
                                return
                            }
                            Log.d(
                                MDNS_LOG_TAG,
                                "Resolving candidate service: ${serviceInfo.toLogSummary()}"
                            )
                            nsdManager.resolveService(
                                serviceInfo,
                                object : NsdManager.ResolveListener {
                                    override fun onServiceResolved(resolved: NsdServiceInfo) {
                                        Log.d(
                                            MDNS_LOG_TAG,
                                            "onServiceResolved: ${resolved.toLogSummary()}"
                                        )
                                        if (!resolved.matchesVi3drServer()) {
                                            Log.d(
                                                MDNS_LOG_TAG,
                                                "Resolved service rejected (name/type mismatch): ${resolved.toLogSummary()}"
                                            )
                                            return
                                        }
                                        val host = resolved.host?.hostAddress
                                        if (host.isNullOrBlank()) {
                                            Log.w(
                                                MDNS_LOG_TAG,
                                                "Resolved service has empty host: ${resolved.toLogSummary()}"
                                            )
                                            return
                                        }
                                        val port = resolved.port
                                        if (port <= 0) {
                                            Log.w(
                                                MDNS_LOG_TAG,
                                                "Resolved service has invalid port=$port: ${resolved.toLogSummary()}"
                                            )
                                            return
                                        }
                                        finish(DiscoveredLanServer(ip = host, port = port))
                                    }

                                    override fun onResolveFailed(
                                        serviceInfo: NsdServiceInfo,
                                        errorCode: Int
                                    ) {
                                        Log.w(
                                            MDNS_LOG_TAG,
                                            "onResolveFailed: errorCode=$errorCode, service=${serviceInfo.toLogSummary()}"
                                        )
                                    }
                                }
                            )
                        }

                        override fun onServiceLost(serviceInfo: NsdServiceInfo) {
                            Log.d(
                                MDNS_LOG_TAG,
                                "onServiceLost: ${serviceInfo.toLogSummary()}"
                            )
                        }

                        override fun onDiscoveryStopped(serviceType: String) {
                            Log.d(MDNS_LOG_TAG, "onDiscoveryStopped: serviceType=$serviceType")
                        }

                        override fun onStartDiscoveryFailed(serviceType: String, errorCode: Int) {
                            Log.e(
                                MDNS_LOG_TAG,
                                "onStartDiscoveryFailed: serviceType=$serviceType, errorCode=$errorCode"
                            )
                            finish(null)
                        }

                        override fun onStopDiscoveryFailed(serviceType: String, errorCode: Int) {
                            Log.e(
                                MDNS_LOG_TAG,
                                "onStopDiscoveryFailed: serviceType=$serviceType, errorCode=$errorCode"
                            )
                            finish(null)
                        }
                    }

                    continuation.invokeOnCancellation {
                        Log.d(
                            MDNS_LOG_TAG,
                            "Discovery cancelled for serviceType=$serviceType; stopping listener."
                        )
                        runCatching { nsdManager.stopServiceDiscovery(discoveryListener) }
                    }

                    runCatching {
                        nsdManager.discoverServices(
                            serviceType,
                            NsdManager.PROTOCOL_DNS_SD,
                            discoveryListener
                        )
                    }.onFailure {
                        Log.e(
                            MDNS_LOG_TAG,
                            "discoverServices threw for serviceType=$serviceType: ${it.message}",
                            it
                        )
                        finish(null)
                    }
                }
            }

            if (discovered != null) {
                Log.i(
                    MDNS_LOG_TAG,
                    "Discovery matched for serviceType=$serviceType -> ip=${discovered.ip}, port=${discovered.port}"
                )
                return discovered
            }
            Log.d(MDNS_LOG_TAG, "No match for serviceType=$serviceType in current attempt.")
        }
        Log.d(MDNS_LOG_TAG, "No matching mDNS service found after all attempts.")
        return null
    }
}

data class DiscoveredLanServer(
    val ip: String,
    val port: Int
)

private fun NsdServiceInfo.matchesVi3drServer(): Boolean {
    val normalizedType = serviceType
        ?.removePrefix(".")
        ?.removeSuffix(".")
        ?.lowercase()

    val expectedType = "_vi3dr._tcp"
    val typeMatches = normalizedType == expectedType
    val nameMatches = serviceName == MDNS_SERVICE_NAME

    return typeMatches && nameMatches
}

private fun NsdServiceInfo.toLogSummary(): String {
    val hostAddress = host?.hostAddress ?: "null"
    return "name=$serviceName, type=$serviceType, host=$hostAddress, port=$port"
}

const val MDNS_SERVICE_TYPE = "_vi3dr._tcp.local."
const val MDNS_SERVICE_NAME = "VI3DR Server"
const val MDNS_LOG_TAG = "WI3DR_MDNS"
val MDNS_DISCOVERY_SERVICE_TYPES = listOf(
    "_vi3dr._tcp.",
    MDNS_SERVICE_TYPE
)
const val MDNS_DISCOVERY_TIMEOUT_MS = 15_000L
