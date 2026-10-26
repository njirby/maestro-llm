// Lets the stock `read` tool "listen": a read of a .wav/.mp3/.flac/.ogg file succeeds with the text
// "Audio read successfully" and attaches the audio, mirroring the stock image path ("Image read
// successfully" + image attachment). On an OpenAI-compatible provider the request then carries
//   {role:"tool", content:"Audio read successfully"}
//   {role:"user", content:[{type:"text",text:"Attached media from tool result:"},
//                          {type:"input_audio",input_audio:{data:<b64>,format:"wav"}}]}
// i.e. exactly where OpenCode puts images read by `read`. See README.md for the captured requests.
//
// Mechanism (OpenCode 1.19.3, no fork; the stock tool schema/description are never touched):
//   tool.execute.before  read on an existing audio file -> point the call's filePath at the file's parent
//                        directory so the stock tool succeeds (a cheap listing) instead of throwing
//                        "Cannot read binary file".
//   tool.execute.after   restore filePath in place (the args object is the tool part's recorded input),
//                        replace the listing with the image-style result and attach the audio.
//   experimental.chat.messages.transform
//                        OpenCode only hoists image/PDF tool attachments into the trailing
//                        "Attached media from tool result:" user message; any other attachment stays in
//                        the tool result, which the openai-compatible provider JSON-stringifies. Hoist the
//                        audio attachments the same way (one user message after each assistant step).
// Every other read (text, directories, images, PDFs, missing files, other binaries) is left untouched.
import type { Plugin } from "@opencode-ai/plugin"
import { execFile } from "node:child_process"
import { mkdtemp, readFile, rm, stat } from "node:fs/promises"
import { tmpdir } from "node:os"
import path from "node:path"

const AUDIO_READ_OUTPUT = "Audio read successfully"
// OpenCode's own label for hoisted tool-result media (constant `t8` in the 1.19.3 binary).
const HOISTED_MEDIA_TEXT = "Attached media from tool result:"

// Mime types the bundled AI SDK openai(-compatible) provider maps to input_audio: audio/wav -> "wav",
// audio/mpeg -> "mp3". Anything else throws UnsupportedFunctionality there, so flac/ogg are decoded to
// 16-bit PCM WAV first (same sample rate and channel count; nothing is resampled or downmixed).
const NATIVE: Record<string, string> = { ".wav": "audio/wav", ".mp3": "audio/mpeg" }
const TRANSCODE = new Set([".flac", ".ogg"])

// Larger files fail the read with an error instead of producing a request the server will reject.
// 64 MiB of base64 audio is ~6 min of 44.1 kHz stereo 16-bit WAV.
const MAX_BYTES = Number(process.env.MAESTRO_AUDIO_READ_MAX_BYTES ?? 64 * 1024 * 1024)

type Pending = { filePath: string; abs: string; ext: string }

// Keyed by `${sessionID}:${callID}`: model-issued call ids are not unique across concurrent sessions.
const pending = new Map<string, Pending>()
const callMessage = new Map<string, string>()
const key = (sessionID: string, callID: string) => `${sessionID}:${callID}`

// Same shape as OpenCode's ascending part ids (prt_ + 12 hex of ms*4096+counter + 14 base62).
let lastMs = 0
let counter = 0
function ascendingID(prefix: "prt" | "msg") {
  const now = Date.now()
  counter = now === lastMs ? counter + 1 : 0
  lastMs = now
  const hex = (BigInt(now) * 4096n + BigInt(counter)).toString(16).padStart(12, "0").slice(-12)
  const chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
  let rand = ""
  for (let i = 0; i < 14; i++) rand += chars[Math.floor(Math.random() * 62)]
  return `${prefix}_${hex}${rand}`
}

const isAudio = (a: { mime: string }) => a.mime.startsWith("audio/")

// Decoded to a temp file, not a pipe: piped WAV output has 0xFFFFFFFF RIFF/data sizes.
async function toWav(abs: string): Promise<Buffer> {
  const dir = await mkdtemp(path.join(tmpdir(), "audio-read-"))
  const out = path.join(dir, "audio.wav")
  try {
    await new Promise<void>((resolve, reject) => {
      execFile(
        "ffmpeg",
        ["-v", "error", "-nostdin", "-i", abs, "-map", "0:a:0", "-c:a", "pcm_s16le", "-map_metadata", "-1", out],
        (err, _stdout, stderr) => {
          if (err) reject(new Error(`Cannot decode audio file: ${abs}\n${String(stderr).trim()}`))
          else resolve()
        },
      )
    })
    return await readFile(out)
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
}

// Only the default export: OpenCode registers every exported function as a plugin.
const AudioRead: Plugin = async ({ client, directory, worktree }) => {
  // The attachment must carry the assistant message id of its tool part, which the hooks do not get.
  // Normally the running-state part event has recorded it; otherwise look the part up.
  async function messageIDFor(sessionID: string, callID: string) {
    const known = callMessage.get(key(sessionID, callID))
    if (known) return known
    const res: any = await client.session.messages({ path: { id: sessionID } })
    for (const msg of res.data ?? []) {
      for (const part of msg.parts ?? []) {
        if (part.type === "tool" && part.callID === callID) return part.messageID as string
      }
    }
    throw new Error(`audio-read: no tool part for call ${callID}`)
  }

  return {
    event: async ({ event }) => {
      if (event.type !== "message.part.updated") return
      const part: any = event.properties.part
      if (part.type !== "tool" || part.tool !== "read") return
      const k = key(part.sessionID, part.callID)
      if (part.state.status === "pending" || part.state.status === "running") callMessage.set(k, part.messageID)
      else callMessage.delete(k)
    },

    "tool.execute.before": async (input, output) => {
      if (input.tool !== "read") return
      const filePath = output.args?.filePath
      if (typeof filePath !== "string") return
      const ext = path.extname(filePath).toLowerCase()
      if (!(ext in NATIVE) && !TRANSCODE.has(ext)) return
      const abs = path.isAbsolute(filePath) ? filePath : path.resolve(directory, filePath)
      const info = await stat(abs).catch(() => undefined)
      if (!info?.isFile()) return // missing file etc.: stock error path
      pending.set(key(input.sessionID, input.callID), { filePath, abs, ext })
      output.args.filePath = path.dirname(abs)
    },

    "tool.execute.after": async (input, output: any) => {
      const k = key(input.sessionID, input.callID)
      const p = pending.get(k)
      if (!p) return
      pending.delete(k)
      input.args.filePath = p.filePath
      try {
        const size = (await stat(p.abs)).size
        if (size > MAX_BYTES) {
          throw new Error(`Audio file too large: ${p.abs} (${size} bytes, limit ${MAX_BYTES})`)
        }
        const mime = NATIVE[p.ext] ?? "audio/wav"
        const bytes = NATIVE[p.ext] ? await readFile(p.abs) : await toWav(p.abs)
        output.title = path.relative(worktree, p.abs) // stock read titles are worktree-relative
        output.output = AUDIO_READ_OUTPUT
        output.metadata = { preview: AUDIO_READ_OUTPUT, truncated: false, loaded: [] }
        output.attachments = [
          {
            type: "file",
            mime,
            url: `data:${mime};base64,${bytes.toString("base64")}`,
            id: ascendingID("prt"),
            sessionID: input.sessionID,
            messageID: await messageIDFor(input.sessionID, input.callID),
          },
        ]
      } finally {
        callMessage.delete(k)
      }
    },

    // Runs on the in-memory history right before OpenCode converts it to provider messages; the stored
    // session is not modified (parts are cloned, never mutated).
    "experimental.chat.messages.transform": async (_input, output) => {
      const msgs = output.messages
      for (let i = 0; i < msgs.length; i++) {
        const msg = msgs[i]
        if (msg.info.role !== "assistant") continue
        const hoisted: any[] = []
        let changed = false
        const parts = msg.parts.map((part: any) => {
          if (part.type !== "tool" || part.state.status !== "completed") return part
          const attachments = part.state.attachments ?? []
          if (!attachments.some(isAudio)) return part
          changed = true
          // Compacted results lose their attachments in OpenCode too.
          if (!part.state.time.compacted) hoisted.push(...attachments.filter(isAudio))
          return { ...part, state: { ...part.state, attachments: attachments.filter((a: any) => !isAudio(a)) } }
        })
        if (!changed) continue
        msgs[i] = { ...msg, parts }
        if (hoisted.length === 0) continue
        const info: any = msg.info
        const id = ascendingID("msg")
        msgs.splice(i + 1, 0, {
          info: {
            id,
            sessionID: info.sessionID,
            role: "user",
            time: { created: info.time.created },
            agent: info.agent ?? info.mode,
            model: { providerID: info.providerID, modelID: info.modelID },
          } as any,
          parts: [
            { id: ascendingID("prt"), sessionID: info.sessionID, messageID: id, type: "text", text: HOISTED_MEDIA_TEXT },
            ...hoisted.map((a) => ({ ...a, id: ascendingID("prt"), messageID: id })),
          ] as any,
        })
        i++
      }
    },
  }
}

export default AudioRead
