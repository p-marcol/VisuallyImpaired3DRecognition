import org.jetbrains.compose.desktop.application.dsl.TargetFormat
import org.gradle.api.tasks.Exec
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidApplication)
    alias(libs.plugins.composeMultiplatform)
    alias(libs.plugins.composeCompiler)
}

kotlin {
    androidTarget {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }
    
    listOf(
        iosArm64(),
        iosSimulatorArm64()
    ).forEach { iosTarget ->
        iosTarget.binaries.framework {
            baseName = "ComposeApp"
            isStatic = true
        }
    }
    
    sourceSets {
        androidMain.dependencies {
            implementation(libs.compose.uiToolingPreview)
            implementation(libs.androidx.activity.compose)
            implementation(libs.ktor.client.okhttp.v340)
            implementation(libs.androidx.camera.core)
            implementation(libs.androidx.camera.camera2)
            implementation(libs.androidx.camera.lifecycle)
            implementation(libs.androidx.camera.view)

        }
        commonMain.dependencies {
            implementation(libs.compose.runtime)
            implementation(libs.compose.foundation)
            implementation(libs.compose.material3)
            implementation(libs.compose.ui)
            implementation(libs.compose.components.resources)
            implementation(libs.compose.uiToolingPreview)
            implementation(libs.androidx.lifecycle.viewmodelCompose)
            implementation(libs.androidx.lifecycle.runtimeCompose)
            implementation(libs.ktor.client.core.v340)
            implementation(libs.ktor.client.websockets.v340)
            implementation(libs.kotlinx.coroutines.core.v1102)

        }
        iosMain.dependencies {
            implementation(libs.ktor.client.darwin.v340)
        }
        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }
    }
}

android {
    namespace = "com.example.wi3dr_kmp"
    compileSdk = libs.versions.android.compileSdk.get().toInt()

    defaultConfig {
        applicationId = "com.example.wi3dr_kmp"
        minSdk = libs.versions.android.minSdk.get().toInt()
        targetSdk = libs.versions.android.targetSdk.get().toInt()
        versionCode = 1
        versionName = "1.0"
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
    buildTypes {
        getByName("release") {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {
    debugImplementation(libs.compose.uiTooling)
}

if (System.getProperty("os.name").lowercase().contains("mac")) {
    fun registerAppleDoubleCleanupTask(
        name: String,
        vararg mustRunAfterTasks: String
    ) = tasks.register<Exec>(name) {
        commandLine(
            "/bin/sh",
            "-c",
            "if [ -d \"\$1\" ]; then /usr/bin/find \"\$1\" -name '._*' -exec /bin/rm -f {} + 2>/dev/null || true; fi",
            "removeAppleDouble",
            layout.buildDirectory.get().asFile.absolutePath
        )
        isIgnoreExitValue = true
        outputs.upToDateWhen { false }
        mustRunAfterTasks.forEach { mustRunAfter(it) }
    }

    val removeAppleDoubleBeforeCommonResourceAccessors =
        registerAppleDoubleCleanupTask("removeAppleDoubleBeforeCommonResourceAccessors")
    val removeAppleDoubleBeforeDebugGeneratedOutputs =
        registerAppleDoubleCleanupTask("removeAppleDoubleBeforeDebugGeneratedOutputs")
    val removeAppleDoubleBeforeDebugKotlinCompile =
        registerAppleDoubleCleanupTask("removeAppleDoubleBeforeDebugKotlinCompile", "processDebugResources")
    val removeAppleDoubleBeforeDebugResourceParsing =
        registerAppleDoubleCleanupTask("removeAppleDoubleBeforeDebugResourceParsing", "packageDebugResources")
    val removeAppleDoubleBeforeDebugResourceLinking =
        registerAppleDoubleCleanupTask(
            "removeAppleDoubleBeforeDebugResourceLinking",
            "mergeDebugResources",
            "parseDebugLocalResources"
        )
    val removeAppleDoubleBeforeDebugDexing =
        registerAppleDoubleCleanupTask(
            "removeAppleDoubleBeforeDebugDexing",
            "compileDebugKotlinAndroid",
            "processDebugResources"
        )
    val removeAppleDoubleBeforeDebugDexMerging =
        registerAppleDoubleCleanupTask("removeAppleDoubleBeforeDebugDexMerging", "dexBuilderDebug")

    listOf(
        "createDebugCompatibleScreenManifests",
        "generateComposeResClass",
        "generateExpectResourceCollectorsForCommonMain",
        "generateResourceAccessorsForCommonMain"
    ).forEach { taskName ->
        tasks.matching { it.name == taskName }.configureEach {
            dependsOn(removeAppleDoubleBeforeDebugGeneratedOutputs)
        }
    }

    tasks.matching { it.name == "generateResourceAccessorsForCommonMain" }.configureEach {
        dependsOn(removeAppleDoubleBeforeCommonResourceAccessors)
    }

    tasks.matching { it.name == "parseDebugLocalResources" }.configureEach {
        dependsOn(removeAppleDoubleBeforeDebugResourceParsing)
    }

    tasks.matching { it.name == "processDebugResources" }.configureEach {
        dependsOn(removeAppleDoubleBeforeDebugResourceLinking)
    }

    tasks.matching { it.name == "compileDebugKotlinAndroid" }.configureEach {
        dependsOn(removeAppleDoubleBeforeDebugKotlinCompile)
    }

    tasks.matching { it.name == "dexBuilderDebug" }.configureEach {
        dependsOn(removeAppleDoubleBeforeDebugDexing)
    }

    tasks.matching { it.name == "mergeProjectDexDebug" }.configureEach {
        dependsOn(removeAppleDoubleBeforeDebugDexMerging)
    }
}
