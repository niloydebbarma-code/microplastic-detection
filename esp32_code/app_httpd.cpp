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
 * - Endpoints: /, /stream, /capture, /status, /control, /restart
 * - Port 81 for MJPEG streaming to Python/YOLOv8
 */

#include "esp_http_server.h"
#include "esp_timer.h"
#include "esp_camera.h"
#include "img_converters.h"
#include "esp32-hal-log.h"
#include "esp_system.h"         // for esp_restart()
#include "freertos/FreeRTOS.h"  // for vTaskDelay
#include "freertos/task.h"      // for portTICK_PERIOD_MS

// MJPEG multipart boundary — must never appear inside JPEG data
#define PART_BOUNDARY "123456789000000000000987654321"
static const char *_STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *_STREAM_BOUNDARY     = "\r\n--" PART_BOUNDARY "\r\n";
static const char *_STREAM_PART         = "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Timestamp: %d.%06d\r\n\r\n";

// stream_httpd port 81 (MJPEG), camera_httpd port 80 (REST) — kept separate so
// the blocking stream loop cannot starve REST requests
httpd_handle_t stream_httpd = NULL;
httpd_handle_t camera_httpd = NULL;

// Forward declarations
static esp_err_t stream_handler(httpd_req_t *req);
static esp_err_t capture_handler(httpd_req_t *req);
static esp_err_t status_handler(httpd_req_t *req);
static esp_err_t control_handler(httpd_req_t *req);
static esp_err_t restart_handler(httpd_req_t *req);
static esp_err_t index_handler(httpd_req_t *req);

// Rolling-average FPS filter
typedef struct {
    size_t size;
    size_t index;
    size_t count;
    int    sum;
    int   *values;
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

// GET /stream — MJPEG multipart stream (port 81, blocking loop)
static esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t    *fb           = NULL;
    struct timeval  _timestamp;
    esp_err_t       res          = ESP_OK;
    size_t          _jpg_buf_len = 0;
    uint8_t        *_jpg_buf     = NULL;
    char            part_buf[128];

    static int64_t last_frame = 0;
    if (!last_frame) {
        last_frame = esp_timer_get_time();
    }

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) {
        return res;
    }
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "X-Framerate", "60");

    while (true) {  // runs until client disconnects
        fb = esp_camera_fb_get();
        if (!fb) {
            log_e("Camera capture failed");
            res = ESP_FAIL;
        } else {
            _timestamp.tv_sec  = fb->timestamp.tv_sec;
            _timestamp.tv_usec = fb->timestamp.tv_usec;

            // OV2640 in PIXFORMAT_JPEG skips conversion; raw/RGB formats go through frame2jpg
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
                _jpg_buf     = fb->buf;
            }
        }

        // boundary → MIME header → JPEG bytes
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        }

        if (res == ESP_OK) {
            size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART,
                                   _jpg_buf_len, _timestamp.tv_sec, _timestamp.tv_usec);
            res = httpd_resp_send_chunk(req, part_buf, hlen);
        }

        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
        }

        // release DMA buffer or heap-allocated JPEG buffer
        if (fb) {
            esp_camera_fb_return(fb);
            fb       = NULL;
            _jpg_buf = NULL;
        } else if (_jpg_buf) {
            free(_jpg_buf);
            _jpg_buf = NULL;
        }

        if (res != ESP_OK) {
            log_e("Send frame failed — closing stream");
            break;
        }

        // log rolling-average FPS to serial
        int64_t fr_end        = esp_timer_get_time();
        int64_t frame_time    = (fr_end - last_frame) / 1000;  // convert µs → ms
        last_frame            = fr_end;
        uint32_t avg_frame_time = ra_filter_run(&ra_filter, (int)frame_time);

        log_i("MJPG: %uB %ums (%.1ffps), AVG: %ums (%.1ffps)",
              (uint32_t)_jpg_buf_len,
              (uint32_t)frame_time,      1000.0 / (uint32_t)frame_time,
              avg_frame_time,            1000.0 / avg_frame_time);
    }

    return res;
}

// GET /capture — single JPEG snapshot
static esp_err_t capture_handler(httpd_req_t *req) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        log_e("Capture failed");
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");

    esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    return res;
}

// GET /control?var=X&val=Y — write OV2640 sensor register at runtime
static esp_err_t control_handler(httpd_req_t *req) {
    char  *buf      = NULL;
    size_t buf_len  = 0;
    char   variable[32] = {0};
    char   value[32]    = {0};

    buf_len = httpd_req_get_url_query_len(req) + 1;
    if (buf_len > 1) {
        buf = (char *)malloc(buf_len);
        if (!buf) {
            httpd_resp_send_500(req);
            return ESP_FAIL;
        }

        if (httpd_req_get_url_query_str(req, buf, buf_len) == ESP_OK) {
            if (httpd_query_key_value(buf, "var", variable, sizeof(variable)) != ESP_OK ||
                httpd_query_key_value(buf, "val", value,    sizeof(value))    != ESP_OK) {
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
    if (!s) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    int res = 0;

    if      (!strcmp(variable, "framesize"))     { if (s->pixformat == PIXFORMAT_JPEG) res = s->set_framesize(s, (framesize_t)val); }
    else if (!strcmp(variable, "quality"))        res = s->set_quality(s, val);
    else if (!strcmp(variable, "contrast"))       res = s->set_contrast(s, val);
    else if (!strcmp(variable, "brightness"))     res = s->set_brightness(s, val);
    else if (!strcmp(variable, "saturation"))     res = s->set_saturation(s, val);
    else if (!strcmp(variable, "gainceiling"))    res = s->set_gainceiling(s, (gainceiling_t)val);
    else if (!strcmp(variable, "colorbar"))       res = s->set_colorbar(s, val);
    else if (!strcmp(variable, "awb"))            res = s->set_whitebal(s, val);
    else if (!strcmp(variable, "agc"))            res = s->set_gain_ctrl(s, val);
    else if (!strcmp(variable, "aec"))            res = s->set_exposure_ctrl(s, val);
    else if (!strcmp(variable, "hmirror"))        res = s->set_hmirror(s, val);
    else if (!strcmp(variable, "vflip"))          res = s->set_vflip(s, val);
    else if (!strcmp(variable, "awb_gain"))       res = s->set_awb_gain(s, val);
    else if (!strcmp(variable, "agc_gain"))       res = s->set_agc_gain(s, val);
    else if (!strcmp(variable, "aec_value"))      res = s->set_aec_value(s, val);
    else if (!strcmp(variable, "aec2"))           res = s->set_aec2(s, val);
    else if (!strcmp(variable, "dcw"))            res = s->set_dcw(s, val);
    else if (!strcmp(variable, "bpc"))            res = s->set_bpc(s, val);
    else if (!strcmp(variable, "wpc"))            res = s->set_wpc(s, val);
    else if (!strcmp(variable, "raw_gma"))        res = s->set_raw_gma(s, val);
    else if (!strcmp(variable, "lenc"))           res = s->set_lenc(s, val);
    else if (!strcmp(variable, "special_effect")) res = s->set_special_effect(s, val);
    else if (!strcmp(variable, "wb_mode"))        res = s->set_wb_mode(s, val);
    else if (!strcmp(variable, "ae_level"))       res = s->set_ae_level(s, val);
    else {
        log_e("Unknown camera control: %s", variable);
        res = -1;
    }

    if (res != 0) {
        log_e("Failed to set %s = %d", variable, val);
        return httpd_resp_send_500(req);
    }

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, NULL, 0);
}

// GET /status — current OV2640 sensor config as JSON
static esp_err_t status_handler(httpd_req_t *req) {
    sensor_t *s = esp_camera_sensor_get();
    if (!s) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    char json_response[512];
    snprintf(json_response, sizeof(json_response),
        "{"
        "\"status\":\"streaming\","
        "\"device\":\"ESP32-CAM AI-Thinker\","
        "\"sensor\":\"OV2640\","
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

// GET /restart — flush response then reboot via esp_restart()
static esp_err_t restart_handler(httpd_req_t *req) {
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_send(req, "Restarting...", 13);
    log_i("Restart requested via /restart endpoint");
    vTaskDelay(500 / portTICK_PERIOD_MS);  // wait for TCP stack to flush the response before rebooting
    esp_restart();
    return ESP_OK;
}

// GET / — embedded single-page UI (raw string, no SPIFFS needed)
static const char INDEX_HTML[] = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Microplastic Detection — ESP32 Camera Control</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg:        #080c10;
  --surface:   #0d1318;
  --card:      #111820;
  --border:    #1e2d3d;
  --accent:    #00e5ff;
  --accent2:   #00ff99;
  --warn:      #ff4f4f;
  --txt:       #c8d8e8;
  --txt-dim:   #4a6070;
  --mono:      'Share Tech Mono', monospace;
  --sans:      'Syne', sans-serif;
  --radius:    6px;
  --glow:      0 0 18px rgba(0,229,255,0.18);
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--txt);
  font-family: var(--sans);
  min-height: 100vh;
  overflow-x: hidden;
}

/* Animated grid background */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(0,229,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
  z-index: 0;
}

/* Top header bar */
header {
  position: relative;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 32px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
}
.logo {
  display: flex;
  align-items: center;
  gap: 12px;
}
.logo-icon {
  width: 36px; height: 36px;
  border: 2px solid var(--accent);
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 16px;
  box-shadow: var(--glow);
  animation: pulse 3s ease-in-out infinite;
}
@keyframes pulse {
  0%,100% { box-shadow: 0 0 10px rgba(0,229,255,0.2); }
  50%      { box-shadow: 0 0 28px rgba(0,229,255,0.55); }
}
.logo-text { font-size: 1.25rem; font-weight: 800; letter-spacing: 0.05em; }
.logo-text span { color: var(--accent); }
.header-meta {
  font-family: var(--mono);
  font-size: 0.7rem;
  color: var(--txt-dim);
  text-align: right;
  line-height: 1.7;
}
.header-meta b { color: var(--accent2); }

/* Status pill */
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: 999px;
  border: 1px solid var(--accent2);
  font-family: var(--mono);
  font-size: 0.7rem;
  color: var(--accent2);
}
.status-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--accent2);
  animation: blink 1.4s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* Nav links */
nav {
  position: relative; z-index: 10;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  padding: 12px 32px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
}
nav a {
  font-family: var(--mono);
  font-size: 0.7rem;
  color: var(--txt-dim);
  text-decoration: none;
  padding: 4px 10px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  transition: all 0.2s;
}
nav a:hover { color: var(--accent); border-color: var(--accent); box-shadow: var(--glow); }

/* Main layout */
main {
  position: relative; z-index: 1;
  display: grid;
  grid-template-columns: 1fr 340px;
  gap: 20px;
  padding: 24px 32px;
  max-width: 1400px;
  margin: 0 auto;
}
@media(max-width:900px) { main { grid-template-columns: 1fr; padding: 16px; } }

/* Card */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  animation: fadein 0.5s ease both;
}
@keyframes fadein { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  background: rgba(0,229,255,0.03);
}
.card-title {
  font-family: var(--mono);
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
}
.card-body { padding: 16px; }

/* Stream viewer */
.stream-wrap {
  position: relative;
  background: #000;
  aspect-ratio: 4/3;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}
.stream-wrap::before {
  content: '';
  position: absolute;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.08) 2px,
    rgba(0,0,0,0.08) 4px
  );
  pointer-events: none;
  z-index: 2;
}
#stream {
  width: 100%; height: 100%;
  object-fit: cover;
  display: block;
}
.stream-overlay {
  position: absolute;
  top: 10px; left: 10px;
  font-family: var(--mono);
  font-size: 0.65rem;
  color: var(--accent2);
  z-index: 3;
  text-shadow: 0 0 8px rgba(0,255,153,0.6);
  pointer-events: none;
}
.stream-overlay-br {
  position: absolute;
  bottom: 10px; right: 10px;
  font-family: var(--mono);
  font-size: 0.65rem;
  color: rgba(0,229,255,0.5);
  z-index: 3;
  pointer-events: none;
}
#stream-status {
  position: absolute;
  font-family: var(--mono);
  font-size: 0.75rem;
  color: var(--txt-dim);
  z-index: 3;
}

/* Action buttons row */
.action-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 12px 16px;
  border-top: 1px solid var(--border);
}
.btn {
  font-family: var(--mono);
  font-size: 0.72rem;
  letter-spacing: 0.05em;
  padding: 7px 14px;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  background: transparent;
  color: var(--txt);
  cursor: pointer;
  transition: all 0.18s;
}
.btn:hover         { border-color: var(--accent); color: var(--accent); box-shadow: var(--glow); }
.btn-accent        { border-color: var(--accent); color: var(--accent); }
.btn-accent:hover  { background: rgba(0,229,255,0.08); }
.btn-success       { border-color: var(--accent2); color: var(--accent2); }
.btn-success:hover { background: rgba(0,255,153,0.08); }
.btn-danger        { border-color: var(--warn); color: var(--warn); }
.btn-danger:hover  { background: rgba(255,79,79,0.1); }

/* Controls panel */
.ctrl-section { margin-bottom: 20px; }
.ctrl-section-title {
  font-family: var(--mono);
  font-size: 0.65rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--txt-dim);
  margin-bottom: 12px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}

/* Sliders */
.ctrl-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
  gap: 10px;
}
.ctrl-label {
  font-family: var(--mono);
  font-size: 0.7rem;
  color: var(--txt);
  min-width: 110px;
  flex-shrink: 0;
}
input[type=range] {
  flex: 1;
  -webkit-appearance: none;
  height: 3px;
  background: var(--border);
  border-radius: 2px;
  outline: none;
  cursor: pointer;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 14px; height: 14px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 8px rgba(0,229,255,0.5);
  cursor: pointer;
  transition: transform 0.15s;
}
input[type=range]::-webkit-slider-thumb:hover { transform: scale(1.3); }
.val-badge {
  font-family: var(--mono);
  font-size: 0.68rem;
  color: var(--accent);
  min-width: 30px;
  text-align: right;
}

/* Select */
select {
  font-family: var(--mono);
  font-size: 0.7rem;
  background: var(--surface);
  color: var(--txt);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 5px 8px;
  cursor: pointer;
  outline: none;
  flex: 1;
  transition: border-color 0.2s;
}
select:focus { border-color: var(--accent); }

/* Toggle switch */
.toggle-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}
.toggle-label {
  font-family: var(--mono);
  font-size: 0.7rem;
  color: var(--txt);
}
.switch { position: relative; display: inline-block; width: 36px; height: 20px; flex-shrink: 0; }
.switch input { opacity: 0; width: 0; height: 0; }
.slider {
  position: absolute; cursor: pointer;
  inset: 0;
  background: var(--border);
  border-radius: 20px;
  transition: 0.3s;
}
.slider:before {
  position: absolute; content: '';
  height: 14px; width: 14px;
  left: 3px; bottom: 3px;
  background: var(--txt-dim);
  border-radius: 50%;
  transition: 0.3s;
}
input:checked + .slider { background: rgba(0,229,255,0.25); border: 1px solid var(--accent); }
input:checked + .slider:before { transform: translateX(16px); background: var(--accent); box-shadow: 0 0 6px rgba(0,229,255,0.6); }

/* Quick-fire buttons grid */
.quick-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
}
.quick-grid .btn { text-align: center; font-size: 0.68rem; }

/* Log panel */
#obs-log {
  list-style: none;
  font-family: var(--mono);
  font-size: 0.68rem;
  max-height: 180px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
#obs-log::-webkit-scrollbar { width: 4px; }
#obs-log::-webkit-scrollbar-track { background: transparent; }
#obs-log::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
#obs-log li {
  padding: 4px 8px;
  border-left: 2px solid var(--border);
  color: var(--txt-dim);
  transition: border-color 0.3s;
}
#obs-log li.new { border-left-color: var(--accent2); color: var(--txt); }
#obs-log li.err { border-left-color: var(--warn); color: #ff8080; }

/* Status JSON */
pre#status-json {
  font-family: var(--mono);
  font-size: 0.68rem;
  color: var(--txt-dim);
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px;
  max-height: 200px;
  overflow-y: auto;
  white-space: pre-wrap;
  line-height: 1.6;
}

/* Debug */
#debug-info {
  font-family: var(--mono);
  font-size: 0.68rem;
  color: var(--txt-dim);
  padding: 8px 0;
}

/* Footer tip */
.tip {
  margin-top: 20px;
  padding: 12px 16px;
  background: rgba(0,229,255,0.04);
  border: 1px solid rgba(0,229,255,0.12);
  border-radius: var(--radius);
  font-family: var(--mono);
  font-size: 0.68rem;
  color: var(--txt-dim);
  line-height: 1.7;
}
.tip code { color: var(--accent2); }

/* Right column scrolls independently */
.right-col { display: flex; flex-direction: column; gap: 16px; }

/* Stagger card animations */
.card:nth-child(1) { animation-delay: 0.05s; }
.card:nth-child(2) { animation-delay: 0.10s; }
.card:nth-child(3) { animation-delay: 0.15s; }
.card:nth-child(4) { animation-delay: 0.20s; }
</style>
</head>
<body>

<!-- ── HEADER ─────────────────────────────────────────────────── -->
<header>
  <div class="logo">
    <div class="logo-icon">&#128300;</div>
    <div>
      <div class="logo-text">Microplastic<span>Detection</span></div>
      <div style="font-family:var(--mono);font-size:0.62rem;color:var(--txt-dim);">ESP32-CAM · OV2640 · Microplastic Detection</div>
    </div>
  </div>
  <div class="header-meta">
    <div>IP &nbsp;<b id="ip">—</b></div>
    <div>Stream &nbsp;<b>:81</b> &nbsp;|&nbsp; REST &nbsp;<b>:80</b></div>
  </div>
  <div class="status-pill"><div class="status-dot"></div><span id="conn-label">LIVE</span></div>
</header>

<!-- ── NAV LINKS ──────────────────────────────────────────────── -->
<nav>
  <a id="stream-link"  href="#" target="_blank">&#9654; STREAM</a>
  <a id="capture-link" href="#" target="_blank">&#9632; CAPTURE</a>
  <a id="status-link"  href="#" target="_blank">&#9675; STATUS JSON</a>
  <a id="control-link" href="#" target="_blank">&#9670; CONTROL API</a>
  <a id="restart-link" href="#" target="_blank">&#9889; RESTART</a>
</nav>

<!-- ── MAIN GRID ──────────────────────────────────────────────── -->
<main>

  <!-- LEFT: stream + controls -->
  <div style="display:flex;flex-direction:column;gap:16px;">

    <!-- Live Feed -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">&#9654; Live Feed</span>
        <span class="status-pill" style="font-size:0.62rem;" id="fps-badge">— fps</span>
      </div>
      <div class="stream-wrap">
        <div class="stream-overlay">&#9679; REC &nbsp; MICROPLASTIC DETECTION v1.0</div>
        <div class="stream-overlay-br" id="res-label">800x600</div>
        <img id="stream" src="" alt="stream">
        <span id="stream-status">Connecting...</span>
      </div>
      <div class="action-row">
        <button class="btn btn-success" onclick="captureSnapshot()">&#9632; Snapshot</button>
        <button class="btn btn-accent"  onclick="location.reload()">&#8635; Refresh</button>
        <button class="btn"             onclick="resetSettings()">&#9881; Defaults</button>
        <button class="btn btn-danger"  onclick="restartESP()">&#9889; Restart</button>
      </div>
    </div>

    <!-- Resolution & Image Quality -->
    <div class="card">
      <div class="card-header"><span class="card-title">&#128208; Resolution &amp; Quality</span></div>
      <div class="card-body">
        <div class="ctrl-section">
          <div class="ctrl-row">
            <span class="ctrl-label">Resolution</span>
            <select id="framesize" onchange="updateValue(this)">
              <option value="5">QVGA 320×240</option>
              <option value="6">VGA 640×480</option>
              <option value="7" selected>SVGA 800×600</option>
              <option value="10">HD 1280×720</option>
              <option value="13">FHD 1920×1080</option>
            </select>
          </div>
          <div class="ctrl-row">
            <span class="ctrl-label">JPEG Quality</span>
            <input type="range" id="quality" min="10" max="63" value="10" onchange="updateValue(this)">
            <span class="val-badge" id="quality-val">10</span>
          </div>
        </div>

        <div class="ctrl-section">
          <div class="ctrl-section-title">Image Adjustments</div>
          <div class="ctrl-row">
            <span class="ctrl-label">Brightness</span>
            <input type="range" id="brightness" min="-2" max="2" value="0" onchange="updateValue(this)">
            <span class="val-badge" id="brightness-val">0</span>
          </div>
          <div class="ctrl-row">
            <span class="ctrl-label">Contrast</span>
            <input type="range" id="contrast" min="-2" max="2" value="0" onchange="updateValue(this)">
            <span class="val-badge" id="contrast-val">0</span>
          </div>
          <div class="ctrl-row">
            <span class="ctrl-label">Saturation</span>
            <input type="range" id="saturation" min="-2" max="2" value="0" onchange="updateValue(this)">
            <span class="val-badge" id="saturation-val">0</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Exposure & White Balance -->
    <div class="card">
      <div class="card-header"><span class="card-title">&#128262; Exposure &amp; White Balance</span></div>
      <div class="card-body">
        <div class="toggle-row"><span class="toggle-label">Auto Exposure (AEC)</span><label class="switch"><input type="checkbox" id="aec" checked onchange="updateValue(this)"><span class="slider"></span></label></div>
        <div class="ctrl-row">
          <span class="ctrl-label">Exposure Val</span>
          <input type="range" id="aec_value" min="0" max="1200" value="300" onchange="updateValue(this)">
          <span class="val-badge" id="aec_value-val">300</span>
        </div>
        <div class="ctrl-row">
          <span class="ctrl-label">AE Level</span>
          <input type="range" id="ae_level" min="-2" max="2" value="0" onchange="updateValue(this)">
          <span class="val-badge" id="ae_level-val">0</span>
        </div>
        <div class="toggle-row"><span class="toggle-label">AEC2 Night Mode</span><label class="switch"><input type="checkbox" id="aec2" onchange="updateValue(this)"><span class="slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">Auto White Balance</span><label class="switch"><input type="checkbox" id="awb" checked onchange="updateValue(this)"><span class="slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">AWB Gain</span><label class="switch"><input type="checkbox" id="awb_gain" checked onchange="updateValue(this)"><span class="slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">Auto Gain (AGC)</span><label class="switch"><input type="checkbox" id="agc" checked onchange="updateValue(this)"><span class="slider"></span></label></div>
        <div class="ctrl-row">
          <span class="ctrl-label">AGC Gain</span>
          <input type="range" id="agc_gain" min="0" max="30" value="0" onchange="updateValue(this)">
          <span class="val-badge" id="agc_gain-val">0</span>
        </div>
        <div class="ctrl-row">
          <span class="ctrl-label">Gain Ceiling</span>
          <select id="gainceiling" onchange="updateValue(this)">
            <option value="0">2x</option><option value="1">4x</option>
            <option value="2" selected>8x</option><option value="3">16x</option>
            <option value="4">32x</option><option value="5">64x</option><option value="6">128x</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Advanced -->
    <div class="card">
      <div class="card-header"><span class="card-title">&#128295; Advanced &amp; Orientation</span></div>
      <div class="card-body">
        <div class="toggle-row"><span class="toggle-label">H-Mirror</span><label class="switch"><input type="checkbox" id="hmirror" onchange="updateValue(this)"><span class="slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">V-Flip</span><label class="switch"><input type="checkbox" id="vflip" onchange="updateValue(this)"><span class="slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">BPC Black Pixel</span><label class="switch"><input type="checkbox" id="bpc" onchange="updateValue(this)"><span class="slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">WPC White Pixel</span><label class="switch"><input type="checkbox" id="wpc" checked onchange="updateValue(this)"><span class="slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">Raw GMA</span><label class="switch"><input type="checkbox" id="raw_gma" checked onchange="updateValue(this)"><span class="slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">Lens Correction</span><label class="switch"><input type="checkbox" id="lenc" checked onchange="updateValue(this)"><span class="slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">DCW Downsize</span><label class="switch"><input type="checkbox" id="dcw" checked onchange="updateValue(this)"><span class="slider"></span></label></div>
        <div class="toggle-row"><span class="toggle-label">Color Bar Test</span><label class="switch"><input type="checkbox" id="colorbar" onchange="updateValue(this)"><span class="slider"></span></label></div>
        <div style="height:10px;"></div>
        <div class="ctrl-row">
          <span class="ctrl-label">Special Effect</span>
          <select id="special_effect" onchange="updateValue(this)">
            <option value="0" selected>None</option><option value="1">Negative</option>
            <option value="2">Grayscale</option><option value="3">Red Tint</option>
            <option value="4">Green Tint</option><option value="5">Blue Tint</option><option value="6">Sepia</option>
          </select>
        </div>
        <div class="ctrl-row">
          <span class="ctrl-label">WB Mode</span>
          <select id="wb_mode" onchange="updateValue(this)">
            <option value="0" selected>Auto</option><option value="1">Sunny</option>
            <option value="2">Cloudy</option><option value="3">Office</option><option value="4">Home</option>
          </select>
        </div>
      </div>
    </div>

  </div><!-- /left col -->

  <!-- RIGHT COLUMN -->
  <div class="right-col">

    <!-- Quick fire -->
    <div class="card">
      <div class="card-header"><span class="card-title">&#9889; Quick Fire</span></div>
      <div class="card-body">
        <div class="quick-grid">
          <button class="btn" onclick="setControl('colorbar',1)">Color Bar ON</button>
          <button class="btn" onclick="setControl('colorbar',0)">Color Bar OFF</button>
          <button class="btn" onclick="setControl('hmirror',1)">Mirror ON</button>
          <button class="btn" onclick="setControl('hmirror',0)">Mirror OFF</button>
          <button class="btn" onclick="setControl('vflip',1)">Flip ON</button>
          <button class="btn" onclick="setControl('vflip',0)">Flip OFF</button>
          <button class="btn" onclick="setControl('aec',1)">Auto Exp ON</button>
          <button class="btn" onclick="setControl('aec',0)">Auto Exp OFF</button>
        </div>
      </div>
    </div>

    <!-- Observation log -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">&#9632; Activity Log</span>
        <button class="btn" style="font-size:0.6rem;padding:3px 8px;" onclick="document.getElementById('obs-log').innerHTML=''">CLR</button>
      </div>
      <div class="card-body" style="padding:12px;">
        <ul id="obs-log"></ul>
      </div>
    </div>

    <!-- Status JSON -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">&#9675; Sensor Status</span>
        <button class="btn btn-accent" style="font-size:0.6rem;padding:3px 8px;" onclick="refreshStatus()">POLL</button>
      </div>
      <div class="card-body" style="padding:12px;">
        <pre id="status-json">Loading...</pre>
        <div id="debug-info" style="margin-top:8px;"></div>
      </div>
    </div>

    <!-- Tip -->
    <div class="tip">
      &#128161; Real-time control — no re-flash needed.<br>
      Python: <code>python 03_esp32_integration.py --esp32 &lt;IP&gt;</code><br>
      Stream: <code>http://&lt;IP&gt;:81/stream</code>
    </div>

  </div><!-- /right col -->

</main>

<script>
const ip         = window.location.hostname;
const httpPort   = '80';
const streamPort = '81';

document.getElementById('ip').textContent = ip;
document.getElementById('stream-link').href  = 'http://' + ip + ':' + streamPort + '/stream';
document.getElementById('capture-link').href = 'http://' + ip + ':' + httpPort   + '/capture';
document.getElementById('status-link').href  = 'http://' + ip + ':' + httpPort   + '/status';
document.getElementById('control-link').href = 'http://' + ip + ':' + httpPort   + '/control?var=framesize&val=7';
document.getElementById('restart-link').href = 'http://' + ip + ':' + httpPort   + '/restart';

var streamImg = document.getElementById('stream');
streamImg.src = 'http://' + ip + ':' + streamPort + '/stream';
document.getElementById('stream-status').style.display = 'none';
streamImg.onerror = function() {
  document.getElementById('stream-status').style.display = 'block';
  document.getElementById('stream-status').textContent   = 'Stream unavailable';
  document.getElementById('conn-label').textContent      = 'OFFLINE';
};

var lastLoad = Date.now(), fpsList = [];
streamImg.onload = function() {
  var now = Date.now(), dt = now - lastLoad; lastLoad = now;
  fpsList.push(1000/dt); if(fpsList.length>10) fpsList.shift();
  var avg = fpsList.reduce(function(a,b){return a+b;},0)/fpsList.length;
  document.getElementById('fps-badge').textContent = avg.toFixed(1) + ' fps';
};

function setControl(varName, val) {
  fetch('http://' + ip + ':' + httpPort + '/control?var=' + varName + '&val=' + val)
    .then(function(r) {
      addLog(r.ok ? ('SET ' + varName + ' = ' + val) : ('ERR ' + varName), !r.ok);
    })
    .catch(function(e) { addLog('NET ERR: ' + e, true); });
}

function updateValue(el) {
  var value = (el.type === 'checkbox') ? (el.checked ? 1 : 0) : el.value;
  var disp  = document.getElementById(el.id + '-val');
  if (disp) disp.textContent = value;
  setControl(el.id, value);
  // update res label if framesize changed
  var resMap = {'5':'320x240','6':'640x480','7':'800x600','10':'1280x720','13':'1920x1080'};
  if (el.id === 'framesize' && resMap[value]) document.getElementById('res-label').textContent = resMap[value];
}

function addLog(msg, isErr) {
  var li   = document.createElement('li');
  var time = new Date().toLocaleTimeString();
  li.textContent = '[' + time + '] ' + msg;
  if (isErr) li.classList.add('err'); else li.classList.add('new');
  var log = document.getElementById('obs-log');
  log.insertBefore(li, log.firstChild);
  while (log.children.length > 30) log.removeChild(log.lastChild);
  setTimeout(function(){ li.classList.remove('new'); }, 2000);
}

function captureSnapshot() {
  window.open('http://' + ip + ':' + httpPort + '/capture', '_blank');
  addLog('Snapshot opened');
}

function restartESP() {
  if (!confirm('Restart ESP32? Stream will disconnect ~5s.')) return;
  fetch('http://' + ip + ':' + httpPort + '/restart')
    .then(function(){ addLog('Restart sent — reloading in 6s...'); })
    .catch(function(){ addLog('Restarting — reloading in 6s...'); });
  setTimeout(function(){ location.reload(); }, 6000);
}

function resetSettings() {
  if (!confirm('Reset all camera settings to defaults?')) return;
  var defaults = {
    framesize:'7', quality:'10', brightness:'0', contrast:'0', saturation:'0',
    ae_level:'0', aec_value:'300', agc_gain:'0',
    aec:true, aec2:false, awb:true, awb_gain:true, agc:true,
    hmirror:false, vflip:false, bpc:false, wpc:true,
    raw_gma:true, lenc:true, dcw:true, colorbar:false,
    special_effect:'0', wb_mode:'0', gainceiling:'2'
  };
  for (var id in defaults) {
    var el = document.getElementById(id); if (!el) continue;
    if (typeof defaults[id] === 'boolean') el.checked = defaults[id];
    else el.value = defaults[id];
    updateValue(el);
  }
  addLog('All settings reset to defaults');
  setTimeout(function(){ location.reload(); }, 1500);
}

function refreshStatus() {
  fetch('http://' + ip + ':' + httpPort + '/status')
    .then(function(r){ return r.json(); })
    .then(function(data) {
      document.getElementById('status-json').textContent = JSON.stringify(data, null, 2);
      document.getElementById('debug-info').textContent  = 'Polled OK';
      var setR = function(id,v){ var e=document.getElementById(id); if(e&&v!==undefined){e.value=v; var d=document.getElementById(id+'-val'); if(d)d.textContent=v;} };
      var setC = function(id,v){ var e=document.getElementById(id); if(e&&v!==undefined) e.checked=(v===1); };
      var setS = function(id,v){ var e=document.getElementById(id); if(e&&v!==undefined) e.value=v; };
      setS('framesize',data.framesize);
      setR('quality',data.quality); setR('brightness',data.brightness);
      setR('contrast',data.contrast); setR('saturation',data.saturation);
      setC('aec',data.aec); setC('awb',data.awb); setC('agc',data.agc);
      setC('hmirror',data.hmirror); setC('vflip',data.vflip);
      addLog('Status synced from ESP32');
    })
    .catch(function(e){
      document.getElementById('status-json').textContent = 'Failed';
      document.getElementById('debug-info').textContent  = 'Error: ' + e;
      addLog('Status poll failed', true);
    });
}

window.onload = function() { refreshStatus(); };
</script>
</body>
</html>
)rawliteral";

static esp_err_t index_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html");
    httpd_resp_set_hdr(req, "Content-Encoding", "identity");
    return httpd_resp_send(req, INDEX_HTML, strlen(INDEX_HTML));
}

// Starts both httpd instances — call once after WiFi connects
void startCameraServer() {
    ra_filter_init(&ra_filter, 20);

    // port 81 — stream only
    httpd_config_t stream_config  = HTTPD_DEFAULT_CONFIG();
    stream_config.server_port     = 81;
    stream_config.ctrl_port       = 32768;

    httpd_uri_t stream_uri = {
        .uri      = "/stream",
        .method   = HTTP_GET,
        .handler  = stream_handler,
        .user_ctx = NULL
    };

    log_i("Starting STREAM server on port 81...");
    if (httpd_start(&stream_httpd, &stream_config) == ESP_OK) {
        httpd_register_uri_handler(stream_httpd, &stream_uri);
        log_i("Stream endpoint ready:  http://<ESP32-IP>:81/stream");
    } else {
        log_e("Failed to start stream server on port 81");
    }

    // port 80 — REST API
    httpd_config_t camera_config  = HTTPD_DEFAULT_CONFIG();
    camera_config.server_port     = 80;
    camera_config.ctrl_port       = 32769;

    httpd_uri_t index_uri = {
        .uri      = "/",
        .method   = HTTP_GET,
        .handler  = index_handler,
        .user_ctx = NULL
    };
    httpd_uri_t capture_uri = {
        .uri      = "/capture",
        .method   = HTTP_GET,
        .handler  = capture_handler,
        .user_ctx = NULL
    };
    httpd_uri_t status_uri = {
        .uri      = "/status",
        .method   = HTTP_GET,
        .handler  = status_handler,
        .user_ctx = NULL
    };
    httpd_uri_t control_uri = {
        .uri      = "/control",
        .method   = HTTP_GET,
        .handler  = control_handler,
        .user_ctx = NULL
    };
    httpd_uri_t restart_uri = {
        .uri      = "/restart",
        .method   = HTTP_GET,
        .handler  = restart_handler,
        .user_ctx = NULL
    };

    log_i("Starting CAMERA server on port 80...");
    if (httpd_start(&camera_httpd, &camera_config) == ESP_OK) {
        httpd_register_uri_handler(camera_httpd, &index_uri);
        httpd_register_uri_handler(camera_httpd, &capture_uri);
        httpd_register_uri_handler(camera_httpd, &status_uri);
        httpd_register_uri_handler(camera_httpd, &control_uri);
        httpd_register_uri_handler(camera_httpd, &restart_uri);
        log_i("Camera server ready.");
        log_i("  Web UI   →  http://<ESP32-IP>:80/");
        log_i("  Capture  →  http://<ESP32-IP>:80/capture");
        log_i("  Status   →  http://<ESP32-IP>:80/status");
        log_i("  Control  →  http://<ESP32-IP>:80/control?var=brightness&val=1");
        log_i("  Restart  →  http://<ESP32-IP>:80/restart");
        log_i("  Stream   →  http://<ESP32-IP>:81/stream");
    } else {
        log_e("Failed to start camera server on port 80");
    }
}