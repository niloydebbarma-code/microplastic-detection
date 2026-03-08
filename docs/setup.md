# Complete Setup Guide for Microplastic Detection System

This guide explains how to set up the entire microplastic detection system step by step. No prior programming experience required.

## About Project

This system automatically detects tiny plastic particles (microplastics) in water samples using:
- **ESP32-CAM**: A small camera board that captures images
- **Python + YOLOv8**: Artificial intelligence that identifies microplastics in the images
- **Your Computer**: Displays results with particle counts

**Think of it as:** A smart microscope that counts plastic particles automatically.

---

## Components

### Hardware
- **ESP32-CAM board** (AI-Thinker model with OV3660 camera)
- **USB programmer** (either ESP32-CAM-MB board or FTDI programmer)
- **USB cable** (Micro-USB)
- **Computer** with Windows, Mac, or Linux
- **WiFi router** OR use ESP32 as its own WiFi hotspot

### Software
- **Arduino IDE** (free software to program ESP32-CAM) - one-time setup
- **Python 3.8 or newer** (free programming language)
- **Your WiFi name and password** (if connecting to router)

---

## Part 1: Setup ESP32-CAM (One-Time, 15 minutes)

This programs the camera to automatically stream video when powered on.

### Step 1: Install Arduino IDE

1. Go to https://www.arduino.cc/en/software
2. Download **Arduino IDE 2.x** for your operating system
3. Install it (click Next → Next → Install)
4. Open Arduino IDE

### Step 2: Add ESP32 Support to Arduino

1. In Arduino IDE, click **File → Preferences**
2. Find the box labeled **"Additional Board Manager URLs"**
3. Paste this link:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
4. Click **OK**
5. Click **Tools → Board → Boards Manager**
6. Type **"esp32"** in the search box
7. Find **"ESP32 by Espressif Systems"** and click **Install**
8. Wait for installation (takes 2-3 minutes)
9. Click **Close**

### Step 3: Prepare the Code

1. Go to the `esp32_code` folder in this project
2. Find the file **`microplastic_detector.ino`**
3. Double-click it to open in Arduino IDE

### Step 4: Enter Your WiFi Information

**Option A: Connect to Your WiFi Router**

1. Find these lines near the top of the code:
   ```cpp
   const char* ssid = "YOUR_WIFI_NAME";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```

2. Change them to your actual WiFi:
   ```cpp
   const char* ssid = "MyHomeWiFi";
   const char* password = "wifi12345";
   ```
   (Use your real WiFi name and password)

**Option B: Make ESP32 Its Own WiFi Hotspot** (No router needed)

1. Find this line:
   ```cpp
   // #define USE_AP_MODE
   ```

2. Remove the `//` to activate it:
   ```cpp
   #define USE_AP_MODE
   ```

3. That's it! ESP32 will create WiFi named **"MicroplasticDetector"** with password **"12345678"**

### Step 5: Configure Arduino Settings

Click these menus in order:

1. **Tools → Board → ESP32 Arduino → AI Thinker ESP32-CAM**
2. **Tools → Upload Speed → 921600**
3. **Tools → Flash Frequency → 80MHz**
4. **Tools → Partition Scheme → Huge APP (3MB No OTA)**

### Step 6: Connect ESP32-CAM to Computer

**If you have ESP32-CAM-MB board (easiest):**
- Insert ESP32-CAM into the MB board
- Connect USB cable from MB board to computer
- That's it!

**If you have FTDI programmer:**
- Connect wires:
  - FTDI **5V** → ESP32 **5V**
  - FTDI **GND** → ESP32 **GND**
  - FTDI **TX** → ESP32 **RX** (labeled U0R)
  - FTDI **RX** → ESP32 **TX** (labeled U0T)
- Connect one more wire: ESP32 **GPIO0** → ESP32 **GND** (this is important for uploading)
- Connect USB cable from FTDI to computer

### Step 7: Select COM Port

1. Click **Tools → Port**
2. Select the port that appeared (like COM3, COM4, or /dev/ttyUSB0)
3. If you don't see any port, check USB cable connection

### Step 8: Upload Code to ESP32-CAM

1. Click the **Upload** button (arrow icon →) in Arduino IDE
2. Wait while it says "Compiling..." and "Uploading..." (takes 1-2 minutes)
3. When you see **"Hard resetting via RTS pin"**, it's done!

**If using FTDI:** Remove the wire between GPIO0 and GND now

4. Press the **RESET** button on ESP32-CAM (small button on the board)

### Step 9: Check If It Worked

1. Click **Tools → Serial Monitor** in Arduino IDE
2. Set the dropdown at bottom right to **115200 baud**
3. Press **RESET** button on ESP32-CAM again
4. You should see:
   ```
   ========================================
   Microplastic Detection System
   ESP32-CAM Initialization
   ========================================
   ✓ PSRAM found - using optimized settings
   ✓ Camera initialized
   ✓ Camera sensor configured
   
   Connecting to WiFi...
   ✓ Connected! IP: 192.168.1.150
   
   ========================================
   System Ready
   ========================================
   Stream URL: http://192.168.1.150:81/stream
   Status URL: http://192.168.1.150:81/status
   ========================================
   ```

5. **Write down the IP address!** (like 192.168.1.150 in the example)
   - If using AP mode, IP will be: **192.168.4.1**

### Step 10: Access Web Control Panel

1. Open your web browser (Chrome, Firefox, etc.)
2. Type the IP address from Serial Monitor (example: `http://192.168.1.150:81`)
3. You should see a **camera control panel** with:
   - **Live video preview** at the top
   - **Control sliders** for brightness, contrast, saturation
   - **Resolution selector** (VGA, SVGA, HD, FHD)
   - **Exposure and white balance** settings
   - **Orientation controls** (mirror, flip)
   - **Special effects** dropdown

4. Try adjusting some settings - changes apply instantly!
5. The stream at `http://192.168.1.150:81/stream` will reflect your changes

**✅ If you see the control panel with live video, ESP32-CAM setup is complete!**

**💡 Tip:** Bookmark this page - you can adjust camera settings anytime without re-uploading code!

---

## Part 2: Setup Python Detection System (10 minutes)

This part sets up the artificial intelligence on your computer.

### Step 1: Check Python Installation

1. Open **Command Prompt** (Windows) or **Terminal** (Mac/Linux)
2. Type this and press Enter:
   ```bash
   python --version
   ```
3. You should see something like "Python 3.8.10" or higher
4. If you get an error, install Python from https://www.python.org/downloads/

### Step 2: Navigate to Project Folder

In the command prompt/terminal, type:
```bash
cd path/to/microplastic_detection
```

Replace `path/to/microplastic_detection` with the actual folder location. Example:
```bash
cd C:\Users\YourName\Downloads\microplastic_detection
```

### Step 3: Install Required Software Packages

Type this command and press Enter:
```bash
pip install ultralytics opencv-python numpy pandas
```

Wait for installation (takes 2-3 minutes). You'll see multiple packages being downloaded.

### Step 4: Verify Model File Exists

Make sure the file **`yolov8_microplastic_trained.pt`** is in the project folder. This is the trained AI model.

If you don't have it, you'll need to train the model first (see Training section below).

---

## Part 3: Running the Detection System

### Basic Usage

1. Make sure ESP32-CAM is powered on and connected to WiFi
2. Open Command Prompt/Terminal in the project folder
3. Run:
   ```bash
   python 03_esp32_integration.py --esp32 YOUR_ESP32_IP
   ```
   
   Replace YOUR_ESP32_IP with the IP from Serial Monitor. Example:
   ```bash
   python 03_esp32_integration.py --esp32 192.168.1.150
   ```
   
   Or if using AP mode:
   ```bash
   python 03_esp32_integration.py --esp32 192.168.4.1
   ```

### What You'll See

A window will open showing:
- **Live camera feed** from ESP32-CAM
- **Green boxes** around detected microplastics
- **Particle count** at the top of screen
- **Detection confidence** for each particle
- **FPS** (frames per second)

Press **'q'** key to stop.

### Advanced Options

**Change detection sensitivity:**
```bash
# More sensitive (detects more particles, may have false positives)
python 03_esp32_integration.py --esp32 192.168.4.1 --conf 0.25

# Balanced (recommended)
python 03_esp32_integration.py --esp32 192.168.4.1 --conf 0.35

# More strict (fewer false positives, might miss some particles)
python 03_esp32_integration.py --esp32 192.168.4.1 --conf 0.50
```

**Save results to video file:**
```bash
python 03_esp32_integration.py --esp32 192.168.4.1 --save
```
Video will be saved as `detection_results.mp4`

**Show processing details:**
```bash
python 03_esp32_integration.py --esp32 192.168.4.1 --verbose
```

---

## Part 4: Training the Model (Optional)

If you want to train your own AI model from scratch:

### Option 1: Using Google Colab (Recommended - Free GPU)

1. Open `colab_train.ipynb` file
2. Go to https://colab.research.google.com/
3. Click **File → Upload notebook**
4. Select the `colab_train.ipynb` file
5. Click **Runtime → Change runtime type → T4 GPU**
6. Click **Runtime → Run all**
7. Wait 2-3 hours for training to complete
8. Download the trained model file `yolov8_microplastic_trained.pt`

### Option 2: On Your Computer (Requires Good GPU)

```bash
# Generate synthetic training data
python 01_generate_synthetic_dataset.py

# Train the model (takes several hours)
python 02_train_yolov8_colab.py
```

Training will create:
- `yolov8_microplastic_trained.pt` - Your trained model
- `confusion_matrix.png` - Shows model accuracy
- `results.png` - Training graphs

---

## Troubleshooting

### ESP32-CAM Issues

**Problem: "Camera init failed: 0x105"**
- Solution: Press RESET button on ESP32-CAM
- Check: Make sure all wires are connected properly
- Check: Board selection is "AI Thinker ESP32-CAM"

**Problem: "WiFi connection failed"**
- Check: WiFi name and password are correct (case-sensitive!)
- Check: ESP32-CAM and computer are on same network
- Try: Use AP mode instead (see Step 4 Option B)

**Problem: "Upload failed" or "Timed out waiting for packet header"**
- Check: GPIO0 is connected to GND during upload (FTDI users)
- Check: Correct COM port selected
- Try: Press and hold BOOT button while uploading
- Try: Lower upload speed to 115200

**Problem: Can't see video in browser**
- Check: Using correct IP address from Serial Monitor
- Check: Including port number :81 in URL
- Try: Disable firewall temporarily
- Try: Use different browser (Chrome recommended)

**Problem: "No module esp_camera.h found"**
- Solution: You didn't install ESP32 board support (go back to Step 2)

### Python Issues

**Problem: "python: command not found"**
- Windows: Try `py` instead of `python`
- Mac/Linux: Install Python from python.org
- Check: Python is added to system PATH

**Problem: "No module named 'ultralytics'"**
- Solution: Run `pip install ultralytics opencv-python`
- Try: `pip3 install ultralytics opencv-python` (on Mac/Linux)

**Problem: "Could not connect to ESP32 stream"**
- Check: ESP32-CAM is powered on
- Check: ESP32 IP address is correct
- Check: Computer is on same WiFi network
- Try: Open stream URL in browser first to verify it works
- Check: Firewall isn't blocking connection

**Problem: "No detections appearing"**
- Check: Model file `yolov8_microplastic_trained.pt` exists
- Try: Lower confidence threshold (--conf 0.25)
- Check: Camera is focused properly
- Check: Holographic sample is visible in stream

**Problem: Detection is too slow/laggy**
- Try: Reduce camera resolution using web UI (switch to VGA 640x480)
- Or: Use web UI at `http://<ESP32_IP>:81` and select lower resolution
- Check: Computer meets minimum requirements
- Close: Other programs using camera or network

### Network Issues

**Problem: Can't ping ESP32 IP address**
- Check: Both devices on same WiFi
- Check: Router allows device-to-device communication
- Try: Use AP mode (ESP32 as hotspot)

**Problem: ESP32 keeps disconnecting**
- Add: Small delay in loop - `delay(100);` in .ino file
- Check: Power supply is stable (use 5V 2A)
- Check: Distance to router (move closer)

---

## System Requirements

### Minimum Requirements
- **Computer:** Windows 7+, macOS 10.14+, or Linux
- **RAM:** 4 GB minimum, 8 GB recommended
- **Python:** 3.8 or newer
- **Internet:** For initial setup and installing packages
- **WiFi:** 2.4 GHz network (ESP32 doesn't support 5 GHz)

### ESP32-CAM Requirements
- **Board:** AI-Thinker ESP32-CAM with OV3660 camera sensor
- **Power:** 5V DC, at least 500mA (1A recommended)
- **WiFi:** 2.4 GHz network or AP mode

---

## Understanding the Results

### What the Green Boxes Mean
- Each green box shows a detected microplastic particle
- **Number on box:** Detection confidence (0-100%)
- **Higher number:** AI is more confident it's a microplastic
- **Lower number:** Might be actual particle or could be noise

### Particle Count
- Shown at top of screen: **"Particles: 12"**
- This is the total count in current frame
- Count updates in real-time as particles move

### Confidence Threshold
- Default is 0.35 (35% confidence required)
- **Lower (0.25):** Detects more particles but may show false positives
- **Higher (0.50):** More accurate but might miss some particles
- **Recommended:** Start with 0.35, adjust based on your needs

### FPS (Frames Per Second)
- Shows processing speed
- **10-15 FPS:** Normal for real-time detection
- **Lower than 10:** Computer might be slow, try closing other programs
- **Higher than 20:** Excellent performance

---

## Tips for Best Results

### Using Web UI Controls (Recommended!)

**Access:** Open `http://<ESP32_IP>:81` in browser (e.g., `http://192.168.4.1:81`)

**Best settings for microplastic detection:**

1. **Resolution:** Start with SVGA (800x600) - good balance of detail and speed
   - Switch to HD (1280x720) if you need more detail
   - Use VGA (640x480) for faster processing

2. **Quality:** Set to 10-15 (lower number = better quality)
   - Lower quality = clearer images but larger file size

3. **Brightness:** Adjust based on your lighting (usually 0 to 1)
   - Too dark: increase brightness
   - Washed out: decrease brightness

4. **Contrast:** Keep at 0 or increase slightly (0 to 1)
   - Helps differentiate particles from background

5. **Exposure:** Keep Auto Exposure ON initially
   - If image flickers, turn OFF and adjust Exposure Value manually

6. **White Balance:** Keep Auto White Balance ON for consistent colors

**💡 Pro Tip:** Use the web UI to find your optimal settings, then they're saved on ESP32-CAM automatically!

### Camera Setup
1. **Lighting:** Use consistent, bright lighting for best detection
2. **Focus:** Ensure camera is properly focused on sample
3. **Distance:** Keep camera at consistent distance from sample
4. **Stability:** Mount ESP32-CAM firmly to avoid shaky video

### Sample Preparation
1. **Water Clarity:** Clearer water gives better results
2. **Concentration:** Moderate particle density works best (not too crowded)
3. **Background:** Plain, uniform background improves detection

### Detection Settings
1. Start with default confidence (0.35)
2. If too many false positives, increase to 0.40 or 0.45
3. If missing particles, decrease to 0.30 or 0.25
4. Use --save option to review results later

---

## Next Steps

Once you have everything working:

1. **Collect Data:** Use --save to record detection videos
2. **Analyze Results:** Review saved videos to verify accuracy
3. **Adjust Settings:** Fine-tune confidence threshold for your samples
4. **Expand Dataset:** Add more training images if needed (see Training section)
5. **Document Findings:** Keep notes on particle counts and water sources

---

## Support and Resources

### Project Files
- **esp32_code/microplastic_detector.ino** - Camera streaming code
- **03_esp32_integration.py** - Detection script
- **yolov8_microplastic_trained.pt** - Trained AI model
- **colab_train.ipynb** - Training notebook

### Additional Help
- Check Serial Monitor for ESP32 status messages
- Use --verbose flag to see detailed processing info
- Review error messages carefully - they usually indicate the problem
- Test each component separately (ESP32 stream, then Python)

### Hardware Sources
- ESP32-CAM AI-Thinker: Available on Amazon, AliExpress, local electronics stores
- Look for: "ESP32-CAM with OV3660" or "AI-Thinker ESP32-CAM"
- ESP32-CAM-MB board: Makes uploading code much easier, recommended

---

## Quick Command Reference

### ESP32-CAM Web Interface

**Main Control Panel:**
```
http://192.168.4.1:81/           # Web UI with all camera controls
http://192.168.4.1:81/stream     # Direct stream (for Python)
http://192.168.4.1:81/status     # JSON status information
```

**Camera Controls Available:**
- Resolution: VGA, SVGA, HD, FHD
- Quality: 10-63 (lower = better)
- Brightness: -2 to 2
- Contrast: -2 to 2
- Saturation: -2 to 2
- Exposure: Auto ON/OFF + manual value
- White Balance: Auto ON/OFF
- Auto Gain: ON/OFF
- Orientation: Mirror/Flip
- Special Effects: None, Grayscale, Sepia, etc.

**💡 All changes apply instantly - no code re-upload needed!**

### Serial Monitor
```bash
# Open Serial Monitor to see ESP32 IP address
# Baud rate: 115200
# Note the IP shown after "System Ready"
```

### Python Detection
```bash
# Basic detection
python 03_esp32_integration.py --esp32 192.168.4.1

# With custom confidence
python 03_esp32_integration.py --esp32 192.168.4.1 --conf 0.35

# Save results
python 03_esp32_integration.py --esp32 192.168.4.1 --save

# Verbose mode
python 03_esp32_integration.py --esp32 192.168.4.1 --verbose

# All options together
python 03_esp32_integration.py --esp32 192.168.4.1 --conf 0.35 --save --verbose
```

### Training (Optional)
```bash
# Generate dataset
python 01_generate_synthetic_dataset.py

# Train model
python 02_train_yolov8_colab.py

# Or use Google Colab (recommended)
# Upload colab_train.ipynb to colab.research.google.com
```

---

## Frequently Asked Questions

**Q: Do I need to upload Arduino code every time?**  
A: No! Upload once, ESP32-CAM remembers it. Just power on and it starts streaming.

**Q: Can I change camera settings without re-uploading code?**  
A: YES! Use the web UI at `http://<ESP32_IP>:81` to adjust all settings in real-time. This is much easier!

**Q: How do I access the camera controls?**  
A: Open `http://192.168.4.1:81` (or your ESP32 IP) in any web browser. You'll see live video and control sliders.

**Q: Can I use this without WiFi?**  
A: Yes! Use AP mode - ESP32 creates its own WiFi hotspot (192.168.4.1).

**Q: What's the difference between web UI and stream endpoint?**  
A: Web UI (`http://IP:81/`) = control panel with settings. Stream endpoint (`http://IP:81/stream`) = raw video for Python.

**Q: How accurate is the detection?**  
A: Model is trained on 15,000+ particles. Accuracy depends on confidence threshold and sample quality.

**Q: Can I detect different particle sizes?**  
A: Yes, trained for 10-500 μm particles. Adjust camera focus for best results.

**Q: Do I need a GPU to run detection?**  
A: No, CPU works fine for real-time detection. GPU only needed for training new models.

**Q: Can I use a different ESP32 or camera?**  
A: This code is specifically for AI-Thinker ESP32-CAM with OV3660. Other models need pin changes.

**Q: How do I know if a detection is correct?**  
A: Higher confidence (>50%) is usually accurate. Save videos and review manually to verify.

**Q: Can I run this on Raspberry Pi?**  
A: Yes! Python code works on Raspberry Pi 4 or newer. ESP32 setup is the same.

---

## Safety Notes

- **Power Supply:** Use stable 5V supply, avoid cheap USB ports that can't provide enough current
- **Heat:** ESP32-CAM may get warm during operation (normal)
- **Water:** Keep electronics away from water samples (obvious but important!)
- **Ventilation:** If device gets very hot, ensure good airflow

---

**You're all set! If you followed all steps, your microplastic detection system should be working.**

For issues not covered here, check error messages carefully - they usually tell you exactly what's wrong.
