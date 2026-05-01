# Vital Plugin Setup (yabridge)

[← Back to README](../README.md)

> **Update (2026-04-30):** Native Linux Vital works correctly for chunk loading. The earlier crashes were caused by `synth_version: "99999.9.9"` in generated presets — Vital's deserializer rejected the version and the resulting state was undefined. See [vital-reaper-gotchas.md](vital-reaper-gotchas.md) for details. yabridge is no longer required but these setup instructions are preserved in case you need it.

yabridge runs Windows Vital via WINE as a fallback. It works but has tradeoffs vs native: 2986 reported params (vs 756 native), param names return as index numbers, and higher per-call overhead.

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
