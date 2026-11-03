// ReinCore desktop app binary entry point.
//
// Prevents additional macOS windows from opening when the user clicks the
// dock icon while the app is already running, then delegates to the library
// `run()` which starts the embedded daemon and the Tauri webview.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    reincore_app_lib::run();
}
