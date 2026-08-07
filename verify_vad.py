import json
import subprocess
import sys
from pathlib import Path

resp = Path("vad_resp.json")
if resp.exists():
    resp.unlink()
F = "data/uploads/c509d8a6bdde4447b9e85945e838fa18.wav"
if not Path(F).is_file():
    # fall back to the newest upload
    uploads = sorted(Path("data/uploads").glob("*.wav"), key=lambda p: p.stat().st_mtime)
    F = str(uploads[-1]) if uploads else F
    print("using newest upload:", F)
r = subprocess.run(
    ["curl", "-s", "-m", "400", "-w", "\nHTTP %{http_code} (%{time_total}s)",
     "-X", "POST", "http://localhost:8000/v1/transcribe", "-F", f"file=@{F}", "-o", str(resp)],
    capture_output=True, text=True,
)
print(r.stdout)
if not resp.exists():
    print("no response file — stderr:", r.stderr[:300])
    sys.exit(1)
d = json.loads(resp.read_text(encoding="utf-8"))
segs = d.get("segments") or []
print("segments:", len(segs), "| chars:", len(d.get("text") or ""))
for s in segs[:3]:
    print("  ", round(s["start"], 1), "-", round(s["end"], 1), "|", s["text"][:70])
print("  ...")
for s in segs[-1:]:
    print("  ", round(s["start"], 1), "-", round(s["end"], 1), "|", s["text"][:70])
resp.unlink(missing_ok=True)
