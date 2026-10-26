#!/usr/bin/env bash
# Render a sound-delta probe dir (jobs.jsonl + presets/) in a throwaway daw-farm container: Xvfb on :99, N processes.
#   maestro/sounddiff/render_in_container.sh outputs/sounddiff/probe0 [N]
set -euo pipefail
DIR=$1; N=${2:-6}
REPO=/home/nate/Documents/maestro-llm
exec docker run --rm --name sounddiff-render --user "$(id -u):$(id -g)" -v /home/nate:/home/nate -w "$REPO" \
  --entrypoint bash daw-farm/reaper:latest -c "
    Xvfb :99 -screen 0 1024x768x24 >/dev/null 2>&1 & sleep 2
    pids=; for s in \$(seq 0 $((N - 1))); do DISPLAY=:99 python3 maestro/sounddiff/render_pairs.py $DIR \$s $N & pids=\"\$pids \$!\"; done
    wait \$pids"
