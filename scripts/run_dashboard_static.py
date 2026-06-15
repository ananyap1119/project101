from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
log = Path("dashboard.detached.bootstrap.log")

try:
    log.write_text("starting dashboard\n", encoding="utf-8")
    dist = ROOT / "dashboard" / "dist"
    handler = partial(SimpleHTTPRequestHandler, directory=str(dist))
    server = ThreadingHTTPServer(("127.0.0.1", 5173), handler)
    server.serve_forever()
except BaseException:
    log.write_text(log.read_text(encoding="utf-8") + traceback.format_exc(), encoding="utf-8")
    raise
