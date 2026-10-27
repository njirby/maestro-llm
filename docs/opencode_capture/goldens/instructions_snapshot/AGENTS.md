This project recreates target sounds as Vital patches with the `vital` skill, which documents the rendering helpers and the patch format. Work in the working directory. You judge sounds by ear only: render, then listen to the audio.

If you're the main session, the user gives you a target sound (target.wav in the working directory). Work like a producer at the desk:
1. Load the `vital` skill first.
2. Listen to the target (read the audio file).
3. Send the target to `transcriber` to get the melody as notes.json.
4. Split the wavetable library into chunks of ~32 tables and send each chunk to a `searcher` subagent in one message, so they run in parallel. Each auditions every table in its chunk and replies with the best 2-3 configurations for the target (tables in oscillator order).
5. Pool the configurations and hand the pool to `judge`, which recombines — one table from one configuration, another from another — and picks the oscillator wavetables.
6. Load those wavetables, then shape the patch subsystem by subsystem: oscillators, envelopes, filter, LFOs, FX, modulation. Render after each change and listen against the target.
7. Stop when another change no longer brings it closer. Say what matches and what still differs.
Write each delegation as plain instructions: the task, why you need it, the exact reply you expect, and the file paths. Subagents only see what you send them.

If you were started as a subagent, do only the task in your prompt and reply as your agent instructions say.
