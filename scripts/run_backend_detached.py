from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
log = Path("backend.detached.bootstrap.log")

try:
    log.write_text("starting backend\n", encoding="utf-8")
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )
except BaseException:
    log.write_text(log.read_text(encoding="utf-8") + traceback.format_exc(), encoding="utf-8")
    raise
