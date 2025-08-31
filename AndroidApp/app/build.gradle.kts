plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    }

    android {
        namespace = "dein.project.mushroomclassifier"
        compileSdk = 34

        defaultConfig {
            applicationId = "dein.project.mushroomclassifier"
            minSdk = 24
            targetSdk = 34
            versionCode = 1
            versionName = "1.0"
        }

        buildTypes {
            release {
                isMinifyEnabled = false
                proguardFiles(
                    getDefaultProguardFile("proguard-android-optimize.txt"),
                    "proguard-rules.pro"
                )
            }
        }
    }

    dependencies {
        implementation("androidx.camera:camera-camera2:1.3.0")
        implementation("androidx.camera:camera-lifecycle:1.3.0")
        implementation("androidx.camera:camera-view:1.3.0")
        implementation("org.tensorflow:tensorflow-lite:2.14.0")
        implementation("com.github.bumptech.glide:glide:4.16.0")
        implementation("com.squareup.retrofit2:retrofit:2.9.0")
    }
}

android {
    namespace = "com.example.mushroomclassification"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.mushroomclassification"
        minSdk = 23
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        viewBinding = true
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.lifecycle.livedata.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.ktx)
    implementation(libs.androidx.navigation.fragment.ktx)
    implementation(libs.androidx.navigation.ui.ktx)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}