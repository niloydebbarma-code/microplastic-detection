/*
 * app_httpd.h - HTTP Server Interface for Microplastic Detection
 */

#ifndef APP_HTTPD_H
#define APP_HTTPD_H

#ifdef __cplusplus
extern "C" {
#endif

// Start the camera HTTP stream server on port 81
void startCameraServer();

#ifdef __cplusplus
}
#endif

#endif // APP_HTTPD_H
