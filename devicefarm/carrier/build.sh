#!/bin/sh
# Builds carrier.apk: a manifest-only package, signed with the local debug key.
#   ANDROID_SDK=<sdk dir> devicefarm/carrier/build.sh   (apksigner needs a JDK on PATH)
set -eu
SDK=${ANDROID_SDK:?set ANDROID_SDK to the Android SDK directory}
BT=$(ls -d "$SDK"/build-tools/* | sort -V | tail -1)
JAR=$(ls -d "$SDK"/platforms/android-36/android.jar)
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HERE/carrier.apk}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
"$BT/aapt2" link -o "$TMP/unsigned.apk" --manifest "$HERE/AndroidManifest.xml" -I "$JAR"
"$BT/zipalign" -f 4 "$TMP/unsigned.apk" "$TMP/aligned.apk"
"$BT/apksigner" sign --ks "$HOME/.android/debug.keystore" --ks-pass pass:android --key-pass pass:android \
  --ks-key-alias androiddebugkey --out "$OUT" "$TMP/aligned.apk"
"$BT/apksigner" verify "$OUT"
echo "$OUT"
