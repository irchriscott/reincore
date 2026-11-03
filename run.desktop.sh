#!/usr/bin/env bash
# Run the ReinCore desktop app.
#
# The desktop app manages the ReinCore local engine as a separate,
# out-of-process daemon. The daemon keeps running after the desktop window
# closes and is reused if it is already serving when the app starts.
#
# Usage:
#   ./run.desktop.sh             # dev mode (hot-reload, opens Tauri window)
#   ./run.desktop.sh --devtools   # dev mode + open DevTools window on launch
#   ./run.desktop.sh --build     # production build (creates installers in target/release/bundle/)
#   ./run.desktop.sh --build-pkg # production build + platform-specific installer (.pkg/.deb/.rpm)
#   ./run.desktop.sh --clean     # remove node_modules and dist before dev

set -euo pipefail

# Resolve the script directory so it can be run from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$SCRIPT_DIR"
APP_DIR="$ENGINE_DIR/app"
FRONTEND_DIR="$APP_DIR/frontend"
RESOURCES_DIR="$APP_DIR/resources"

# Detect the platform-specific daemon binary name.
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
  PLATFORM="windows"
  DAEMON_BIN_NAME="reincore.exe"
elif [[ "$OSTYPE" == "darwin"* ]]; then
  PLATFORM="macos"
  DAEMON_BIN_NAME="reincore"
else
  PLATFORM="linux"
  DAEMON_BIN_NAME="reincore"
fi

# Parse the mode flags.
MODE="dev"
CLEAN=false
BUILD_PKG=false
OPEN_DEVTOOLS=false
for arg in "$@"; do
  case "$arg" in
    --build)     MODE="build" ;;
    --build-pkg) MODE="build"; BUILD_PKG=true ;;
    --clean)     CLEAN=true ;;
    --devtools)  OPEN_DEVTOOLS=true ;;
  esac
done


# Optional: clean build artifacts before starting.
if [ "$CLEAN" = true ]; then
  echo "Cleaning node_modules and dist..."
  rm -rf "$FRONTEND_DIR/node_modules" "$FRONTEND_DIR/dist"
fi

# Ensure frontend dependencies are installed.
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "Installing frontend dependencies..."
  cd "$FRONTEND_DIR"
  npm install
fi

# Ensure the generated icons exist (Tauri build fails without them).
if [ ! -f "$APP_DIR/icons/icon.ico" ] && [ ! -f "$APP_DIR/icons/icon.icns" ] && [ ! -f "$APP_DIR/icons/32x32.png" ]; then
  echo "Generating Tauri icon set from source.png..."
  cd "$FRONTEND_DIR"
  npx @tauri-apps/cli icon "$APP_DIR/icons/source.png" -o "$APP_DIR/icons"
fi

# Run the Tauri CLI from the frontend's node_modules/.bin so it resolves
# correctly on Windows (Git Bash).
TAURI_BIN="$FRONTEND_DIR/node_modules/.bin/tauri"

if [ "$MODE" = "build" ]; then
  echo "Building ReinCore CLI binaries (reincore + rc)..."
  cd "$ENGINE_DIR"
  cargo build --release --bin reincore --bin rc

  echo "Bundling CLI binaries into Tauri resources..."
  mkdir -p "$RESOURCES_DIR"
  cp "$ENGINE_DIR/target/release/$DAEMON_BIN_NAME" "$RESOURCES_DIR/$DAEMON_BIN_NAME"
  if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    cp "$ENGINE_DIR/target/release/rc.exe" "$RESOURCES_DIR/rc.exe"
  else
    cp "$ENGINE_DIR/target/release/rc" "$RESOURCES_DIR/rc"
  fi

  echo "Building ReinCore desktop app (production)..."
  echo "  Output:    target/release/bundle/"
  echo ""
  cd "$APP_DIR"
  "$TAURI_BIN" build

  if [ "$BUILD_PKG" = true ]; then
    echo ""
    echo "Building platform-specific installer..."
    case "$PLATFORM" in
      macos)
        "$ENGINE_DIR/packaging/macos/build_pkg.sh"
        ;;
      linux)
        if command -v dpkg-deb >/dev/null 2>&1; then
          "$ENGINE_DIR/packaging/linux/build_cli_deb.sh"
        fi
        if command -v rpmbuild >/dev/null 2>&1; then
          "$ENGINE_DIR/packaging/linux/build_cli_rpm.sh"
        fi
        ;;
      windows)
        echo "Windows installers are already produced by Tauri (NSIS/MSI)."
        ;;
    esac
  fi
else
  echo "Starting ReinCore desktop app (dev)..."
  echo "  Frontend:  http://localhost:1420"
  echo "  Daemon:    http://127.0.0.1:8989"
  if [ "$OPEN_DEVTOOLS" = true ]; then
    echo "  DevTools:  enabled (RENCORE_DEVTOOLS=1)"
    export RENCORE_DEVTOOLS=1
  else
    echo "  DevTools:  disabled (use --devtools to enable)"
  fi
  echo ""
  cd "$APP_DIR"
  "$TAURI_BIN" dev
fi
