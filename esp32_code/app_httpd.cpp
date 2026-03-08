/*
 * app_httpd.cpp - HTTP Server for Microplastic Detection System
 * Source: ESP32 CameraWebServer example (Espressif Systems)
 * 
 * Original Copyright 2015-2016 Espressif Systems (Shanghai) PTE LTD
 * Licensed under the Apache License, Version 2.0
 * 
 * Modifications for Microplastic Detection:
 * - Removed face detection/recognition features
 * - Removed LED flash control  
 * - Web UI with camera controls
 * - Endpoints: /, /stream, /status, /control
 * - Port 81 for MJPEG streaming to Python/YOLOv8
 * - Optimized for lensless digital holography
 */

#include "esp_http_server.h"
#include "esp_timer.h"
#include "esp_camera.h"
#include "img_converters.h"

// MJPEG Stream boundary
#define PART_BOUNDARY "123456789000000000000987654321"
static const char *_STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *_STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *_STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Timestamp: %d.%06d\r\n\r\n";

// HTTP server handles
httpd_handle_t stream_httpd = NULL;
httpd_handle_t camera_httpd = NULL;

// FPS filter for logging
typedef struct {
    size_t size;
    size_t index;
    size_t count;
    int sum;
    int *values;
} ra_filter_t;

static ra_filter_t ra_filter;

static ra_filter_t *ra_filter_init(ra_filter_t *filter, size_t sample_size) {
    memset(filter, 0, sizeof(ra_filter_t));
    filter->values = (int *)malloc(sample_size * sizeof(int));
    if (!filter->values) {
        return NULL;
    }
    memset(filter->values, 0, sample_size * sizeof(int));
    filter->size = sample_size;
    return filter;
}

static int ra_filter_run(ra_filter_t *filter, int value) {
    if (!filter->values) {
        return value;
    }
    filter->sum -= filter->values[filter->index];
    filter->values[filter->index] = value;
    filter->sum += filter->values[filter->index];
    filter->index++;
    filter->index = filter->index % filter->size;
    if (filter->count < filter->size) {
        filter->count++;
    }
    return filter->sum / filter->count;
}

// ============================================================================
// HTTP Handler: /stream - MJPEG Stream for YOLOv8 Detection
// ============================================================================
static esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t *fb = NULL;
    struct timeval _timestamp;
    esp_err_t res = ESP_OK;
    size_t _jpg_buf_len = 0;
    uint8_t *_jpg_buf = NULL;
    char part_buf[128];

    static int64_t last_frame = 0;
    if (!last_frame) {
        last_frame = esp_timer_get_time();
    }

    // Set MJPEG multipart content type
    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) {
        return res;
    }

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "X-Framerate", "60");

    // Streaming loop
    while (true) {
        // Capture frame from camera
        fb = esp_camera_fb_get();
        if (!fb) {
            log_e("Camera capture failed");
            res = ESP_FAIL;
        } else {
            _timestamp.tv_sec = fb->timestamp.tv_sec;
            _timestamp.tv_usec = fb->timestamp.tv_usec;

            // Convert to JPEG if needed
            if (fb->format != PIXFORMAT_JPEG) {
                bool jpeg_converted = frame2jpg(fb, 80, &_jpg_buf, &_jpg_buf_len);
                esp_camera_fb_return(fb);
                fb = NULL;
                if (!jpeg_converted) {
                    log_e("JPEG compression failed");
                    res = ESP_FAIL;
                }
            } else {
                _jpg_buf_len = fb->len;
                _jpg_buf = fb->buf;
            }
        }

        // Send multipart boundary
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        }

        // Send JPEG header with content length and timestamp
        if (res == ESP_OK) {
            size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, 
                                   _jpg_buf_len, _timestamp.tv_sec, _timestamp.tv_usec);
            res = httpd_resp_send_chunk(req, part_buf, hlen);
        }

        // Send JPEG data
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
        }

        // Clean up frame buffer
        if (fb) {
            esp_camera_fb_return(fb);
            fb = NULL;
            _jpg_buf = NULL;
        } else if (_jpg_buf) {
            free(_jpg_buf);
            _jpg_buf = NULL;
        }

        // Check for errors
        if (res != ESP_OK) {
            log_e("Send frame failed");
            break;
        }

        // Calculate and log FPS
        int64_t fr_end = esp_timer_get_time();
        int64_t frame_time = (fr_end - last_frame) / 1000;  // Convert to ms
        last_frame = fr_end;
        uint32_t avg_frame_time = ra_filter_run(&ra_filter, frame_time);
        
        log_i("MJPG: %uB %ums (%.1ffps), AVG: %ums (%.1ffps)",
              (uint32_t)_jpg_buf_len,
              (uint32_t)frame_time, 1000.0 / (uint32_t)frame_time,
              avg_frame_time, 1000.0 / avg_frame_time);
    }

    return res;
}

// ============================================================================
// HTTP Handler: /control - Camera Settings Control
// ============================================================================
static esp_err_t control_handler(httpd_req_t *req) {
    char *buf = NULL;
    size_t buf_len = 0;
    char variable[32] = {0,};
    char value[32] = {0,};

    buf_len = httpd_req_get_url_query_len(req) + 1;
    if (buf_len > 1) {
        buf = (char*)malloc(buf_len);
        if (!buf) {
            httpd_resp_send_500(req);
            return ESP_FAIL;
        }
        if (httpd_req_get_url_query_str(req, buf, buf_len) == ESP_OK) {
            if (httpd_query_key_value(buf, "var", variable, sizeof(variable)) == ESP_OK &&
                httpd_query_key_value(buf, "val", value, sizeof(value)) == ESP_OK) {
            } else {
                free(buf);
                httpd_resp_send_404(req);
                return ESP_FAIL;
            }
        } else {
            free(buf);
            httpd_resp_send_404(req);
            return ESP_FAIL;
        }
        free(buf);
    } else {
        httpd_resp_send_404(req);
        return ESP_FAIL;
    }

    int val = atoi(value);
    log_i("Camera control: %s = %d", variable, val);
    sensor_t *s = esp_camera_sensor_get();
    int res = 0;

    if (!s) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    // Process camera controls
    if (!strcmp(variable, "framesize")) {
        if (s->pixformat == PIXFORMAT_JPEG) res = s->set_framesize(s, (framesize_t)val);
    }
    else if (!strcmp(variable, "quality")) res = s->set_quality(s, val);
    else if (!strcmp(variable, "contrast")) res = s->set_contrast(s, val);
    else if (!strcmp(variable, "brightness")) res = s->set_brightness(s, val);
    else if (!strcmp(variable, "saturation")) res = s->set_saturation(s, val);
    else if (!strcmp(variable, "gainceiling")) res = s->set_gainceiling(s, (gainceiling_t)val);
    else if (!strcmp(variable, "colorbar")) res = s->set_colorbar(s, val);
    else if (!strcmp(variable, "awb")) res = s->set_whitebal(s, val);
    else if (!strcmp(variable, "agc")) res = s->set_gain_ctrl(s, val);
    else if (!strcmp(variable, "aec")) res = s->set_exposure_ctrl(s, val);
    else if (!strcmp(variable, "hmirror")) res = s->set_hmirror(s, val);
    else if (!strcmp(variable, "vflip")) res = s->set_vflip(s, val);
    else if (!strcmp(variable, "awb_gain")) res = s->set_awb_gain(s, val);
    else if (!strcmp(variable, "agc_gain")) res = s->set_agc_gain(s, val);
    else if (!strcmp(variable, "aec_value")) res = s->set_aec_value(s, val);
    else if (!strcmp(variable, "aec2")) res = s->set_aec2(s, val);
    else if (!strcmp(variable, "dcw")) res = s->set_dcw(s, val);
    else if (!strcmp(variable, "bpc")) res = s->set_bpc(s, val);
    else if (!strcmp(variable, "wpc")) res = s->set_wpc(s, val);
    else if (!strcmp(variable, "raw_gma")) res = s->set_raw_gma(s, val);
    else if (!strcmp(variable, "lenc")) res = s->set_lenc(s, val);
    else if (!strcmp(variable, "special_effect")) res = s->set_special_effect(s, val);
    else if (!strcmp(variable, "wb_mode")) res = s->set_wb_mode(s, val);
    else if (!strcmp(variable, "ae_level")) res = s->set_ae_level(s, val);
    else {
        log_e("Unknown camera control: %s", variable);
        res = -1;
    }

    if (res) {
        log_e("Failed to set %s to %d", variable, val);
        return httpd_resp_send_500(req);
    }

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, NULL, 0);
}

// ============================================================================
// HTTP Handler: /status - System Status JSON
// ============================================================================
static esp_err_t status_handler(httpd_req_t *req) {
    sensor_t *s = esp_camera_sensor_get();
    if (!s) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    // Build JSON status response
    char json_response[512];
    snprintf(json_response, sizeof(json_response),
        "{"
        "\"status\":\"streaming\","
        "\"device\":\"ESP32-CAM AI-Thinker\","
        "\"sensor\":\"OV3660\","
        "\"project\":\"Microplastic Detection\","
        "\"framesize\":%d,"
        "\"quality\":%d,"
        "\"brightness\":%d,"
        "\"contrast\":%d,"
        "\"saturation\":%d,"
        "\"awb\":%d,"
        "\"aec\":%d,"
        "\"hmirror\":%d,"
        "\"vflip\":%d"
        "}",
        s->status.framesize,
        s->status.quality,
        s->status.brightness,
        s->status.contrast,
        s->status.saturation,
        s->status.awb,
        s->status.aec,
        s->status.hmirror,
        s->status.vflip
    );

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json_response, strlen(json_response));
}

// ============================================================================
// HTTP Handler: / - Web UI Control Panel
// ============================================================================
static const char PROGMEM INDEX_HTML[] = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Microplastic Detection - Camera Control</title>
    <style>
        body {web UI endpoint
    httpd_uri_t index_uri = {
        .uri       = "/",
        .method    = HTTP_GET,
        .handler   = index_handler,
        .user_ctx  = NULL
    };

    // Define stream endpoint
    httpd_uri_t stream_uri = {
        .uri       = "/stream",
        .method    = HTTP_GET,
        .handler   = stream_handler,
        .user_ctx  = NULL
    };

    // Define status endpoint
    httpd_uri_t status_uri = {
        .uri       = "/status",
        .method    = HTTP_GET,
        .handler   = status_handler,
        .user_ctx  = NULL
    };

    // Define control endpoint
    httpd_uri_t control_uri = {
        .uri       = "/control",
        .method    = HTTP_GET,
        .handler   = control_handler,
        .user_ctx  = NULL
    };

    // Start HTTP server
    log_i("Starting camera server on port: %d", config.server_port);
    if (httpd_start(&stream_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(stream_httpd, &index_uri);
        httpd_register_uri_handler(stream_httpd, &stream_uri);
        httpd_register_uri_handler(stream_httpd, &status_uri);
        httpd_register_uri_handler(stream_httpd, &control_uri);
        log_i("Camera server started successfully");
        log_i("Web UI: http://<ESP32-IP>:81/");
        log_i("Stream: http://<ESP32-IP>:81/stream");
        log_i("Status: http://<ESP32-IP>:81/status");
        log_i("Control: http://<ESP32-IP>:81/control
<body>
    <div class="container">
        <h1>🔬 Microplastic Detection Camera</h1>
        
        <div class="info">
            <strong>Device:</strong> ESP32-CAM AI-Thinker | <strong>Sensor:</strong> OV3660 | <strong>Stream:</strong> <a href="/stream" target="_blank">http://<span id="ip"></span>:81/stream</a>
        </div>

        <div class="stream-section">
            <h2>Live Camera Feed</h2>
            <img id="stream" src="/stream" style="display:none;" onload="this.style.display='block';">
        </div>

        <div class="btn-group">
            <button class="btn" onclick="location.reload()">🔄 Refresh</button>
            <button class="btn" onclick="resetSettings()">⚙️ Reset to Defaults</button>
        </div>

        <div class="controls">
            <div class="control-group">
                <h3>📐 Resolution & Quality</h3>
                <label>Resolution:
                    <select id="framesize" onchange="updateValue(this)">
                        <option value="5">QVGA (320x240)</option>
                        <option value="6">VGA (640x480)</option>
                        <option value="7" selected>SVGA (800x600)</option>
                        <option value="10">HD (1280x720)</option>
                        <option value="13">FHD (1920x1080)</option>
                    </select>
                </label>
                <label>Quality (0-63, lower=better):
                    <input type="range" id="quality" min="10" max="63" value="10" onchange="updateValue(this)">
                    <span class="value-display" id="quality-val">10</span>
                </label>
            </div>

            <div class="control-group">
                <h3>🎨 Image Adjustments</h3>
                <label>Brightness (-2 to 2):
                    <input type="range" id="brightness" min="-2" max="2" value="0" onchange="updateValue(this)">
                    <span class="value-display" id="brightness-val">0</span>
                </label>
                <label>Contrast (-2 to 2):
                    <input type="range" id="contrast" min="-2" max="2" value="0" onchange="updateValue(this)">
                    <span class="value-display" id="contrast-val">0</span>
                </label>
                <label>Saturation (-2 to 2):
                    <input type="range" id="saturation" min="-2" max="2" value="0" onchange="updateValue(this)">
                    <span class="value-display" id="saturation-val">0</span>
                </label>
            </div>

            <div class="control-group">
                <h3>🔆 Exposure & White Balance</h3>
                <label>Auto Exposure:
                    <label class="switch"><input type="checkbox" id="aec" checked onchange="updateValue(this)"><span class="slider"></span></label>
                </label>
                <label>Exposure Value (0-1200):
                    <input type="range" id="aec_value" min="0" max="1200" value="300" onchange="updateValue(this)">
                    <span class="value-display" id="aec_value-val">300</span>
                </label>
                <label>Auto White Balance:
                    <label class="switch"><input type="checkbox" id="awb" checked onchange="updateValue(this)"><span class="slider"></span></label>
                </label>
                <label>Auto Gain:
                    <label class="switch"><input type="checkbox" id="agc" checked onchange="updateValue(this)"><span class="slider"></span></label>
                </label>
            </div>

            <div class="control-group">
                <h3>🔄 Orientation</h3>
                <label>Horizontal Mirror:
                    <label class="switch"><input type="checkbox" id="hmirror" onchange="updateValue(this)"><span class="slider"></span></label>
                </label>
                <label>Vertical Flip:
                    <label class="switch"><input type="checkbox" id="vflip" onchange="updateValue(this)"><span class="slider"></span></label>
                </label>
            </div>

            <div class="control-group">
                <h3>🔧 Advanced</h3>
                <label>Special Effect:
                    <select id="special_effect" onchange="updateValue(this)">
                        <option value="0" selected>None</option>
                        <option value="1">Negative</option>
                        <option value="2">Grayscale</option>
                        <option value="3">Red Tint</option>
                        <option value="4">Green Tint</option>
                        <option value="5">Blue Tint</option>
                        <option value="6">Sepia</option>
                    </select>
                </label>
                <label>Color Bar Test:
                    <label class="switch"><input type="checkbox" id="colorbar" onchange="updateValue(this)"><span class="slider"></span></label>
                </label>
            </div>
        </div>

        <div style="text-align:center; margin-top:30px; color:#666; font-size:12px;">
            <p>💡 <strong>Tip:</strong> Adjust settings in real-time without re-uploading code!</p>
            <p>Use with Python detection: <code>python 03_esp32_integration.py --esp32 &lt;IP&gt;</code></p>
        </div>
    </div>

    <script>
        // Set IP address in page
        document.getElementById('ip').textContent = window.location.hostname;

        function updateValue(el) {
            let value = el.type === 'checkbox' ? (el.checked ? 1 : 0) : el.value;
            let id = el.id;
            
            // Update display value
            let valDisplay = document.getElementById(id + '-val');
            if (valDisplay) valDisplay.textContent = value;
            
            // Send to ESP32
            fetch(`/control?var=${id}&val=${value}`)
                .then(response => {
                    if (!response.ok) console.error('Failed to update', id);
                })
                .catch(err => console.error('Error:', err));
        }

        function resetSettings() {
            if (confirm('Reset all camera settings to defaults?')) {
                // Reset all controls
                document.getElementById('framesize').value = '7';
                document.getElementById('quality').value = '10';
                document.getElementById('brightness').value = '0';
                document.getElementById('contrast').value = '0';
                document.getElementById('saturation').value = '0';
                document.getElementById('aec').checked = true;
                document.getElementById('aec_value').value = '300';
                document.getElementById('awb').checked = true;
                document.getElementById('agc').checked = true;
                document.getElementById('hmirror').checked = false;
                document.getElementById('vflip').checked = false;
                document.getElementById('special_effect').value = '0';
                document.getElementById('colorbar').checked = false;
                
                // Apply all
                document.querySelectorAll('input, select').forEach(el => updateValue(el));
                
                setTimeout(() => location.reload(), 1000);
            }
        }

        // Load current settings on page load
        window.onload = function() {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    if (data.framesize !== undefined) document.getElementById('framesize').value = data.framesize;
                    if (data.quality !== undefined) {
                        document.getElementById('quality').value = data.quality;
                        document.getElementById('quality-val').textContent = data.quality;
                    }
                    if (data.brightness !== undefined) {
                        document.getElementById('brightness').value = data.brightness;
                        document.getElementById('brightness-val').textContent = data.brightness;
                    }
                    if (data.contrast !== undefined) {
                        document.getElementById('contrast').value = data.contrast;
                        document.getElementById('contrast-val').textContent = data.contrast;
                    }
                    if (data.saturation !== undefined) {
                        document.getElementById('saturation').value = data.saturation;
                        document.getElementById('saturation-val').textContent = data.saturation;
                    }
                    if (data.aec !== undefined) document.getElementById('aec').checked = data.aec === 1;
                    if (data.awb !== undefined) document.getElementById('awb').checked = data.awb === 1;
                    if (data.hmirror !== undefined) document.getElementById('hmirror').checked = data.hmirror === 1;
                    if (data.vflip !== undefined) document.getElementById('vflip').checked = data.vflip === 1;
                })
                .catch(err => console.error('Failed to load settings:', err));
        };
    </script>
</body>
</html>
)rawliteral";

static esp_err_t index_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html");
    httpd_resp_set_hdr(req, "Content-Encoding", "identity");
    return httpd_resp_send(req, INDEX_HTML, strlen(INDEX_HTML));
}

// ============================================================================
// Start Camera Server on Port 81
// ============================================================================
void startCameraServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 81;  // Port 81 for Python integration
    config.ctrl_port = 32768;

    // Initialize FPS filter
    ra_filter_init(&ra_filter, 20);

    // Define stream endpoint
    httpd_uri_t stream_uri = {
        .uri       = "/stream",
        .method    = HTTP_GET,
        .handler   = stream_handler,
        .user_ctx  = NULL
    };

    // Define status endpoint
    httpd_uri_t status_uri = {
        .uri       = "/status",
        .method    = HTTP_GET,
        .handler   = status_handler,
        .user_ctx  = NULL
    };

    // Start HTTP server
    log_i("Starting camera server on port: %d", config.server_port);
    if (httpd_start(&stream_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(stream_httpd, &stream_uri);
        httpd_register_uri_handler(stream_httpd, &status_uri);
        log_i("Camera server started successfully");
        log_i("Stream endpoint: http://<ESP32-IP>:81/stream");
        log_i("Status endpoint: http://<ESP32-IP>:81/status");
    } else {
        log_e("Failed to start camera server");
    }
}
