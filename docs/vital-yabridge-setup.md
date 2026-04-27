# Vital Plugin Setup (yabridge)

[← Back to README](../README.md)

**Vital's Linux-native builds (VST2/VST3/CLAP, all versions through 1.6.0) crash REAPER** when loading preset state containing any real wavetable. The crash fires inside Vital's own state deserializer — it's not a transport issue and can't be worked around via chunked writes or file-handoff. Any `VitalController.set_preset()` call with a non-trivial wavetable will segfault REAPER.

**Workaround:** run Vital as a Windows plugin bridged through [yabridge](https://github.com/robbert-vdh/yabridge) + WINE. This is a **hard dependency** for live-exec grading (`scripts/grade_agent_sft.py --live-exec-check`) and for any agent rollout that applies wavetables to a live REAPER session.

```bash
# 1. WINE staging
sudo dpkg --add-architecture i386
sudo mkdir -pm755 /etc/apt/keyrings
sudo wget -O /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key
sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/noble/winehq-noble.sources
sudo apt update
sudo apt install --install-recommends winehq-staging

# 2. yabridge (user-local)
mkdir -p ~/.local/share/yabridge ~/.local/bin
# Download latest yabridge release tarball from https://github.com/robbert-vdh/yabridge/releases
tar xzf yabridge-*.tar.gz -C ~/.local/share/yabridge --strip-components=1
ln -sf ~/.local/share/yabridge/yabridgectl ~/.local/bin/yabridgectl

# 3. Install Windows Vital via WINE (requires VitalInstaller.exe from vital.audio)
#    In the installer component screen, check: VST3, VST, CLAP
wine ~/VitalInstaller.exe

# 4. Bridge the plugins into Linux plugin paths
yabridgectl add "$HOME/.wine/drive_c/Program Files/Common Files/VST3"
yabridgectl add "$HOME/.wine/drive_c/Program Files/Common Files/CLAP"
yabridgectl add "$HOME/.wine/drive_c/Program Files/Steinberg/VstPlugins"
yabridgectl sync

# 5. Move aside any Linux-native Vital so REAPER picks the bridged one
for d in ~/.vst3/Vital.vst3 ~/.vst/Vital.so ~/.clap/Vital.clap; do
    [ -e "$d" ] && mv "$d" "$d.linux-native-bak"
done
```

Restart REAPER; it will scan and register the bridged plugins under the usual "Vital (Vital Audio)" names.
