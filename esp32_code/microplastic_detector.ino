/*
 * Microplastic Detection System - ESP32-CAM
 * Lensless Digital Holography for Water Quality Monitoring
 * 
 * Hardware: ESP32-CAM (AI-Thinker) with OV2640 2MP sensor
 * Purpose: Stream holographic images to Python YOLOv8 detection server
 * 
 * Configuration:
 * - Board: AI Thinker ESP32-CAM
 * - Upload Speed: 921600
 * - Flash Frequency: 80MHz
 * - Partition Scheme: Huge APP (3MB No OTA)
 * 
 * Project: Automated Microplastic Detection and Quantification using Deep Learning
 * Repository: https://github.com/niloydebbarma-code/microplastic-detection
 */

#include "esp_camera.h"
#include <WiFi.h>
#include "camera_pins.h"  // Pin definitions for AI-Thinker board
#include "app_httpd.h"    // HTTP server for streaming

// ============================================================================
// WiFi Configuration (Change these!)
// ============================================================================
const char* ssid = "YOUR_WIFI_NAME";        // Your WiFi SSID
const char* password = "YOUR_WIFI_PASSWORD"; // Your WiFi password

// Alternative: Create Access Point (uncomment to use)
// Default AP IP: 192.168.4.1 (ESP32 standard)
// const char* ssid = "MicroplasticDetector";
// const char* password = "12345678";
// #define USE_AP_MODE  // Uncomment this line to create WiFi hotspot

// ============================================================================
// Setup Function
// ============================================================================
// ============================================================================
// Setup Function
// ============================================================================
void setup() {
    Serial.begin(115200);
    Serial.setDebugOutput(true);
    Serial.println();
    Serial.println("========================================");
    Serial.println("Microplastic Detection System");
    Serial.println("ESP32-CAM Initialization");
    Serial.println("========================================");

    // ========================================================================
    // Camera Configuration (AI-Thinker with OV2640)
    // ========================================================================
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.grab_mode = CAMERA_GRAB_LATEST;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.jpeg_quality = 12;
    config.fb_count = 1;

    // Frame size and quality optimized for holography
    if (psramFound()) {
        config.frame_size = FRAMESIZE_SVGA;  // 800x600 for holography
        config.jpeg_quality = 10;             // 0-63 lower is better quality
        config.fb_count = 2;
        config.grab_mode = CAMERA_GRAB_LATEST;
        Serial.println("✓ PSRAM found - using optimized settings");
    } else {
        config.frame_size = FRAMESIZE_SVGA;
        config.fb_location = CAMERA_FB_IN_DRAM;
        Serial.println("⚠ PSRAM not found - using DRAM");
    }

    // Initialize Camera
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("✗ Camera init failed: 0x%x\n", err);
        return;
    }
    Serial.println("✓ Camera initialized");

    // ========================================================================
    // Camera Sensor Settings (OV2640 optimized for holography)
    // ========================================================================
    sensor_t * s = esp_camera_sensor_get();
    s->set_brightness(s, 0);     // -2 to 2
    s->set_contrast(s, 0);       // -2 to 2
    s->set_saturation(s, 0);     // -2 to 2
    s->set_special_effect(s, 0); // 0: No Effect
    s->set_whitebal(s, 1);       // White balance
    s->set_awb_gain(s, 1);       // Auto white balance gain
    s->set_wb_mode(s, 0);        // White balance mode
    s->set_exposure_ctrl(s, 1);  // Auto exposure
    s->set_aec2(s, 0);           // Auto exposure correction
    s->set_ae_level(s, 0);       // -2 to 2
    s->set_aec_value(s, 300);    // 0 to 1200
    s->set_gain_ctrl(s, 1);      // Auto gain
    s->set_agc_gain(s, 0);       // 0 to 30
    s->set_gainceiling(s, (gainceiling_t)0);
    s->set_bpc(s, 0);            // Black pixel correction
    s->set_wpc(s, 1);            // White pixel correction
    s->set_raw_gma(s, 1);        // Raw gamma
    s->set_lenc(s, 1);           // Lens correction
    s->set_hmirror(s, 0);        // Horizontal flip
    s->set_vflip(s, 0);          // Vertical flip
    s->set_dcw(s, 1);            // Downsize enable
    s->set_colorbar(s, 0);       // Color bar test pattern
    Serial.println("✓ Camera sensor configured");

    // ========================================================================
    // WiFi Setup
    // ========================================================================
    #ifdef USE_AP_MODE
        // Create Access Point (Default IP: 192.168.4.1)
        Serial.println("\nCreating WiFi Access Point...");
        WiFi.softAP(ssid, password);
        IPAddress IP = WiFi.softAPIP();
        Serial.print("✓ AP IP address: ");
        Serial.println(IP);
        Serial.println("   (Default ESP32 AP IP: 192.168.4.1)");
        Serial.printf("✓ SSID: %s\n", ssid);
        Serial.printf("✓ Password: %s\n", password);
    #else
        // Connect to WiFi
        Serial.println("\nConnecting to WiFi...");
        WiFi.begin(ssid, password);
        WiFi.setSleep(false);
        
        int attempts = 0;
        while (WiFi.status() != WL_CONNECTED && attempts < 20) {
            delay(500);
            Serial.print(".");
            attempts++;
        }
        
        if (WiFi.status() == WL_CONNECTED) {
            Serial.println();
            Serial.print("✓ Connected! IP: ");
            Serial.println(WiFi.localIP());
        } else {
            Serial.println("\n✗ WiFi connection failed");
            return;
        }
    #endif

    // ========================================================================
    // Start Camera Server (from app_httpd.cpp)
    // ========================================================================
    startCameraServer();

    Serial.println("\n========================================");
    Serial.println("System Ready");
    Serial.println("========================================");
    #ifdef USE_AP_MODE
        Serial.printf("Web UI:     http://%s:80/\n", WiFi.softAPIP().toString().c_str());
        Serial.printf("Stream URL: http://%s:81/stream\n", WiFi.softAPIP().toString().c_str());
        Serial.printf("Status URL: http://%s:80/status\n", WiFi.softAPIP().toString().c_str());
    #else
        Serial.printf("Web UI:     http://%s:80/\n", WiFi.localIP().toString().c_str());
        Serial.printf("Stream URL: http://%s:81/stream\n", WiFi.localIP().toString().c_str());
        Serial.printf("Status URL: http://%s:80/status\n", WiFi.localIP().toString().c_str());
    #endif
    Serial.println("========================================");
    Serial.println("💡 Open Web UI in browser to control camera!");
    Serial.println("========================================");
}

// ============================================================================
// Main Loop
// ============================================================================

void loop() {
    // Main loop does nothing - server runs in background
    delay(10000);
}