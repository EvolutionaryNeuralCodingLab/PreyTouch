#!/bin/bash

# https://peter.sh/experiments/chromium-command-line-switches/
# "--ignore-certificate-errors"  

params=( "--disable-session-crashed-bubble" "--disable-infobars" "--disable-component-update" \
         "--disable-pinch" "--chrome-frame" "--window-size=$2"  "--window-position=$4,001" \
         "--remember-cert-error-decisions" "--no-default-browser-check" "--no-first-run" \
         "--ignore-urlfetcher-cert-requests" "--allow-running-insecure-content" \
         '--simulate-outdated-no-au="01 Jan 2199"' \
         "--display=$3" )

case "$*" in
(*--kiosk*) params+=( "--kiosk" );;
esac

/opt/google/chrome/google-chrome "${params[@]}" http://localhost:8080/#/$1 &
echo google-chrome "${params[@]}" http://localhost:8080/#/$1