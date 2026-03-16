# ESP32-CAM Code for Microplastic Detection

This folder contains the Arduino code for ESP32-CAM camera module.

## Files

- **microplastic_detector.ino** - Main Arduino sketch (open this in Arduino IDE)
- **app_httpd.cpp** - HTTP server for video streaming
- **app_httpd.h** - HTTP server header file
- **camera_pins.h** - Pin definitions for AI-Thinker board
- **camera_index.h** - Web interface (optional, not used)

## Setup Instructions

**📖 For complete setup instructions, see: [docs/setup.md](../docs/setup.md)**

## Quick Start

1. Install Arduino IDE with ESP32 support
2. Open `microplastic_detector.ino`
3. Configure your WiFi name and password (lines 24-25)
4. Select board: AI Thinker ESP32-CAM
5. Upload to ESP32-CAM
6. Check Serial Monitor for IP address
7. **Open web UI:** `http://YOUR_ESP32_IP:81/` to control camera
8. Use stream with Python: `python 03_esp32_integration.py --esp32 <IP>`

## Web UI Control Panel

Open `http://<ESP32_IP>:81/` in your browser (example: `http://192.168.4.1:81/`):

✨ **Features:**
- 📹 **Live camera preview** - real-time video feed
- 🎛️ **Resolution selector** - VGA (640x480), SVGA (800x600), HD (1280x720), FHD (1920x1080)
- ⚙️ **Image controls** - brightness, contrast, saturation sliders
- 🔆 **Exposure & White Balance** - auto/manual modes, exposure value adjustment
- 🔄 **Orientation tools** - horizontal mirror, vertical flip toggles
- 🎨 **Special effects** - grayscale, sepia, negative, color tints
- ⚡ **Instant apply** - all changes happen in real-time, no code re-upload!
- 🔁 **Reset button** - restore default settings with one click

**Why use Web UI?**
- ✅ Adjust settings without editing code
- ✅ Test different configurations instantly
- ✅ Perfect for demos and presentations
- ✅ Non-programmers can control camera
- ✅ Find optimal settings for your water samples

## Features

- **Web UI Control Panel:** Adjust all camera settings in real-time via browser
- **MJPEG Streaming:** `/stream` endpoint for Python/OpenCV integration
- **Status API:** `/status` endpoint returns JSON with current settings
- **Real-time Adjustments:** Change brightness, contrast, resolution without re-uploading code

## Endpoints

- `http://<ESP32_IP>:81/` - **Web control panel** (adjust settings)
- `http://<ESP32_IP>:81/stream` - MJPEG video stream
- `http://<ESP32_IP>:81/status` - JSON status info
- `http://<ESP32_IP>:81/control?var=brightness&val=1` - API control

## Default Settings

- **WiFi Mode:** Station (connects to your router)
- **AP Mode IP:** 192.168.4.1 (if using access point mode)
- **Stream Port:** 81
- **Stream URL:** `http://<ESP32_IP>:81/stream`
- **Status URL:** `http://<ESP32_IP>:81/status`

## Arduino IDE Settings

- Board: **AI Thinker ESP32-CAM**
- Upload Speed: **115200**
- Flash Frequency: **80MHz**
- Partition Scheme: **Huge APP (3MB No OTA)**

## Need Help?

See the complete guide: **[docs/setup.md](../docs/setup.md)**
