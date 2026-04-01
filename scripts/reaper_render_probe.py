#!/usr/bin/env python3
"""Render the first track in REAPER to /tmp/probe.wav via reapy.

Usage:
    python scripts/reaper_render_probe.py

Solos track 0, renders the full project to /tmp/probe.wav, then unsolos.
Requires REAPER running with the reapy server active.
"""
import reapy

with reapy.inside_reaper():
    api = reapy.reascript_api
    track_id = reapy.Project().tracks[0].id
    api.SetMediaTrackInfo_Value(track_id, "I_SOLO", 2)
    api.GetSetProjectInfo_String(0, "RENDER_FILE", "/tmp", True)
    api.GetSetProjectInfo_String(0, "RENDER_PATTERN", "probe", True)
    api.GetSetProjectInfo(0, "RENDER_SETTINGS", 0, True)
    api.GetSetProjectInfo(0, "RENDER_BOUNDSFLAG", 1, True)
    api.Main_OnCommand(41824, 0)
    api.SetMediaTrackInfo_Value(track_id, "I_SOLO", 0)

print("Rendered to /tmp/probe.wav")
