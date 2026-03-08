"""
ESP32-CAM Microplastic Detection System

Real-time microplastic detection using lensless digital holography.
Integrates trained YOLOv8 model with ESP32-CAM for automated water quality monitoring.

Usage:
    python 03_esp32_integration.py
    python 03_esp32_integration.py --esp32 192.168.1.100 --model yolov8_microplastic_trained.pt
"""

import cv2
from ultralytics import YOLO
from flask import Flask, Response, render_template_string
import argparse
import time
from pathlib import Path

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Microplastic Detection System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 3px solid #0f4c75;
        }
        h1 {
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d9ff, #0099ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #aaa;
            font-size: 1.1em;
        }
        .info-bar {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            gap: 10px;
            flex-wrap: wrap;
        }
        .info-box {
            background: rgba(15, 76, 117, 0.3);
            padding: 10px 20px;
            border-radius: 10px;
            border: 1px solid #0f4c75;
        }
        .info-label { color: #888; font-size: 0.9em; }
        .info-value { color: #00d9ff; font-weight: bold; font-size: 1.1em; }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .stat-card {
            background: rgba(15, 76, 117, 0.3);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            border: 2px solid #0f4c75;
            transition: transform 0.3s, border-color 0.3s;
        }
        .stat-card:hover {
            transform: translateY(-5px);
            border-color: #00d9ff;
        }
        .stat-value {
            font-size: 3em;
            font-weight: bold;
            background: linear-gradient(90deg, #00d9ff, #0099ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.2;
        }
        .stat-label {
            color: #aaa;
            margin-top: 10px;
            font-size: 1.1em;
        }
        .video-container {
            background: rgba(15, 76, 117, 0.2);
            padding: 20px;
            border-radius: 20px;
            border: 2px solid #0f4c75;
            margin: 20px 0;
        }
        .video-frame {
            width: 100%;
            border-radius: 10px;
            border: 3px solid #0f4c75;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #888;
            border-top: 1px solid #0f4c75;
        }
        .status-online {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #00ff00;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        .water-status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }
        .safe { background: #27ae60; color: white; }
        .unsafe { background: #e74c3c; color: white; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
    <script>
        // Auto-refresh stats every 2 seconds
        setInterval(function() {
            fetch('/stats')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('count').textContent = data.count;
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('time').textContent = data.runtime;
                    
                    // Update water quality status
                    const status = data.count < 10 ? 'SAFE' : 'UNSAFE';
                    const statusClass = data.count < 10 ? 'safe' : 'unsafe';
                    const statusEl = document.getElementById('water-status');
                    statusEl.textContent = status;
                    statusEl.className = 'water-status ' + statusClass;
                })
                .catch(err => console.error('Stats update failed:', err));
        }, 2000);
    </script>
</head>
<body>
    <div class="container">
        <header>
            <h1>� Microplastic Detection System</h1>
            <p class="subtitle">Real-Time Water Quality Monitoring via Digital Holography</p>
        </header>
        
        <div class="info-bar">
            <div class="info-box">
                <div class="info-label">Status</div>
                <div class="info-value"><span class="status-online"></span> ONLINE</div>
            </div>
            <div class="info-box">
                <div class="info-label">ESP32-CAM</div>
                <div class="info-value">{{ esp32_ip }}</div>
            </div>
            <div class="info-box">
            <div class="info-box">
                <div class="info-label">Water Quality</div>
                <div class="info-value"><span id="water-status" class="water-status safe">SAFE</span></div>
            </div>
                <div class="info-label">Model</div>
                <div class="info-value">YOLOv8n</div>
            </div>
            <div class="info-box">
                <div class="info-label">Confidence</div>
                <div class="info-value">{{ confidence }}</div>
            </div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value" id="count">{{ count }}</div>
                <div class="stat-label">Microplastics Detected</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="fps">{{ fps }}</div>
                <div class="stat-label">Frames Per Second</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="time">00:00</div>
                <div class="stat-label">Runtime</div>
            </div>
        </div>
        
        <div class="video-container">
            <img class="video-frame" src="/video_feed" alt="Live Detection Feed">
        </div>
        
        <div class="footer">
            <p><strong>Lensless Digital Holography for Water Quality</strong></p>
            <p>Automated Microplastic Detection & Quantification</p>
            <p style="margin-top: 10px; color: #666;">
                Hardware: ESP32-CAM (OV3660) + 650nm Laser | AI: YOLOv8-nano | SDG 6.3
            </p>
            <p style="margin-top: 5px; color: #555;">
                Detection Range: 10μm+ | Training: HMPD Dataset (15,106 patches)
            </p>
        </div>
    </div>
</body>
</html>
"""

# ===== APPLICATION CLASS =====
class MicroplasticDetectionApp:
    """Flask-based real-time microplastic detection via lensless holography."""
    
    def __init__(self, esp32_ip, model_path, confidence, port=5000):
        self.esp32_ip = esp32_ip
        self.esp32_stream = f"http://{esp32_ip}:81/stream"
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.port = port
        
        self.detection_count = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        
        self.model = self._load_model()
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _load_model(self):
        """Load YOLOv8 model."""
        print("\nLoading model...")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            model = YOLO(str(self.model_path))
            print(f"Model loaded: {self.model_path}")
            return model
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def _setup_routes(self):
        """Configure Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template_string(
                HTML_TEMPLATE,
                esp32_ip=self.esp32_ip,
                model_path=self.model_path.name,
                confidence=self.confidence,
                count=self.detection_count,
                fps=self.fps
            )
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @self.app.route('/stats')
        def stats():
            runtime = time.time() - self.start_time
            mins, secs = divmod(int(runtime), 60)
            return {
                'count': self.detection_count,
                'fps': self.fps,
                'runtime': f"{mins:02d}:{secs:02d}"
            }
    
    def _generate_frames(self):
        """Stream video frames with detections."""
        cap = cv2.VideoCapture(self.esp32_stream)
        
        if not cap.isOpened():
            print(f"\nCannot connect to ESP32-CAM: {self.esp32_stream}")
            return
        
        print(f"Connected to ESP32-CAM: {self.esp32_stream}")
        
        frame_times = []
        
        while True:
            success, frame = cap.read()
            
            if not success:
                time.sleep(0.1)
                continue
            
            start = time.time()
            results = self.model(frame, conf=self.confidence, verbose=False)
            inference_time = time.time() - start
            
            boxes = results[0].boxes
            self.detection_count += len(boxes)
            self.frame_count += 1
            
            frame_times.append(time.time())
            if len(frame_times) > 30:
                frame_times.pop(0)
            if len(frame_times) > 1:
                self.fps = len(frame_times) / (frame_times[-1] - frame_times[0])
            
            annotated = results[0].plot()
            
            cv2.putText(annotated, f"Microplastics: {len(boxes)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Total Count: {self.detection_count}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"FPS: {self.fps:.1f}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', annotated)
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    def run(self):
        """Start Flask server."""
        print("\n" + "=" * 70)
        print("Microplastic Detection System - Lensless Holography")
        print("=" * 70)
        print(f"ESP32-CAM:  {self.esp32_stream}")
        print(f"Model:      {self.model_path.name}")
        print(f"Confidence: {self.confidence}")
        print(f"Dashboard:  http://localhost:{self.port}")
        print("=" * 70)
        print("\nSystem ready. Press Ctrl+C to stop.\n")
        
        self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Microplastic Detection via ESP32-CAM + YOLOv8')
    parser.add_argument('--esp32', type=str, default='192.168.1.100',
                        help='ESP32-CAM IP address (default: 192.168.1.100)')
    parser.add_argument('--model', type=str, default='yolov8_microplastic_trained.pt',
                        help='Path to trained model (default: yolov8_microplastic_trained.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Dashboard port (default: 5000)')
    
    args = parser.parse_args()
    
    try:
        app = MicroplasticDetectionApp(
            esp32_ip=args.esp32,
            model_path=args.model,
            confidence=args.conf,
            port=args.port
        )
        app.run()
    except KeyboardInterrupt:
        print("\n\nSystem stopped")
    except Exception as e:
        print(f"\nError: {e}")
        import sys
        sys.exit(1)
