#!/bin/sh
# Builds carrier.apk: a manifest-only package, signed with the local debug key.
#   ANDROID_SDK=<sdk dir> devicefarm/carrier/build.sh   (apksigner needs a JDK on PATH)
set -eu
SDK=${ANDROID_SDK:?set ANDROID_SDK to the Android SDK directory}
# Pinned, so two builds of this APK use the same tools; install them with sdkmanager if missing.
BUILD_TOOLS=${BUILD_TOOLS:-37.0.0}
PLATFORM=${PLATFORM:-android-36}
BT="$SDK/build-tools/$BUILD_TOOLS"
JAR="$SDK/platforms/$PLATFORM/android.jar"
for tool in "$BT/aapt2" "$BT/zipalign" "$BT/apksigner" "$JAR"; do
  [ -e "$tool" ] || { echo "missing $tool (sdkmanager 'build-tools;$BUILD_TOOLS' 'platforms;$PLATFORM')" >&2; exit 1; }
done
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HERE/carrier.apk}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
"$BT/aapt2" link -o "$TMP/unsigned.apk" --manifest "$HERE/AndroidManifest.xml" -I "$JAR"
"$BT/zipalign" -f 4 "$TMP/unsigned.apk" "$TMP/aligned.apk"
"$BT/apksigner" sign --ks "$HOME/.android/debug.keystore" --ks-pass pass:android --key-pass pass:android \
  --ks-key-alias androiddebugkey --out "$OUT" "$TMP/aligned.apk"
"$BT/apksigner" verify "$OUT"
echo "$OUT"
