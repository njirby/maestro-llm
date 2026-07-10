#!/bin/bash
# Sample vLLM concurrency metrics every 5s until killed or $1 samples.
N="${1:-240}"
URLS="${2:-http://localhost:8000}"
for i in $(seq 1 "$N"); do
    line="$(date +%s)"
    for u in $URLS; do
        m="$(curl -s -m 3 "$u/metrics" 2>/dev/null)"
        run="$(echo "$m" | grep '^vllm:num_requests_running' | awk '{print $2}' | paste -sd+ - | bc 2>/dev/null)"
        wait="$(echo "$m" | grep '^vllm:num_requests_waiting' | awk '{print $2}' | paste -sd+ - | bc 2>/dev/null)"
        line="$line ${u##*:}:run=${run:-NA},wait=${wait:-NA}"
    done
    echo "$line"
    sleep 5
done
