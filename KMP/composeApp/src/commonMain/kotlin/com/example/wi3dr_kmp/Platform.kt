package com.example.wi3dr_kmp

interface Platform {
    val name: String
}

expect fun getPlatform(): Platform