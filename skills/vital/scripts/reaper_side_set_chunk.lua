-- ReaScript (Lua): read a VST chunk from disk, write it into a track's FX.
-- Invoked from outside REAPER when the encoded chunk is too large to pass
-- through the Python/C string binding (~256 KiB ReaScript crash threshold).
--
-- Invocation protocol (see maestro/reaper/vital_tools.py::_set_preset_via_file):
--   /tmp/vital_chunk_req.json  — {chunk_path, track_idx, fx_idx, parm_name}
--   <chunk_path>               — encoded base64 string
--   /tmp/vital_chunk_done.json — status output

local REQ_PATH  = "/tmp/vital_chunk_req.json"
local DONE_PATH = "/tmp/vital_chunk_done.json"

local function read_file(path)
  local f, err = io.open(path, "rb")
  if not f then return nil, err end
  local data = f:read("*a")
  f:close()
  return data
end

local function write_status(tbl)
  local f = io.open(DONE_PATH, "w")
  if not f then return end
  local parts = {}
  for k, v in pairs(tbl) do
    if type(v) == "string" then
      parts[#parts + 1] = string.format("%q: %q", k, v)
    else
      parts[#parts + 1] = string.format("%q: %s", k, tostring(v))
    end
  end
  f:write("{" .. table.concat(parts, ", ") .. "}")
  f:close()
end

-- Minimal JSON-ish parser for the request file. The request is a flat object
-- with string and integer values; we pull the specific fields by regex.
local function parse_req(txt)
  local req = {}
  req.chunk_path = txt:match('"chunk_path"%s*:%s*"([^"]+)"')
  req.parm_name  = txt:match('"parm_name"%s*:%s*"([^"]+)"') or "vst_chunk"
  local ti = txt:match('"track_idx"%s*:%s*(%-?%d+)')
  local fi = txt:match('"fx_idx"%s*:%s*(%-?%d+)')
  req.track_idx = ti and tonumber(ti) or 0
  req.fx_idx    = fi and tonumber(fi) or 0
  return req
end

local ok, err = pcall(function()
  local req_txt, read_err = read_file(REQ_PATH)
  if not req_txt then error("could not read req: " .. tostring(read_err)) end
  local req = parse_req(req_txt)
  if not req.chunk_path then error("req missing chunk_path") end

  local encoded, chunk_err = read_file(req.chunk_path)
  if not encoded then error("could not read chunk: " .. tostring(chunk_err)) end

  local track = reaper.GetTrack(0, req.track_idx)
  if not track then error("GetTrack(" .. req.track_idx .. ") returned nil") end

  local ret = reaper.TrackFX_SetNamedConfigParm(track, req.fx_idx, req.parm_name, encoded)
  if ret == false then error("TrackFX_SetNamedConfigParm returned false") end

  write_status({status = "ok", bytes = #encoded})
  os.remove(req.chunk_path)
end)

if not ok then
  write_status({status = "error", msg = tostring(err):sub(1, 400)})
end
