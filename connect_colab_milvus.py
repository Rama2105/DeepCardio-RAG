"""
connect_colab_milvus.py
=======================
Reads milvus_tunnel.json written by the Colab notebook and patches .env
so the local backend connects to Milvus running on Google Colab via ngrok.

Usage:
    python connect_colab_milvus.py            # update .env only
    python connect_colab_milvus.py --restart  # update .env + restart backend
    python connect_colab_milvus.py --reset    # restore localhost (after Colab session ends)
"""

import json
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent
TUNNEL_FILE = BASE / "milvus_tunnel.json"
ENV_FILE = BASE / ".env"


def read_tunnel() -> dict:
    if not TUNNEL_FILE.exists():
        print("[ERROR] milvus_tunnel.json not found.")
        print("Make sure the Colab ngrok cell has run and Google Drive is synced.")
        sys.exit(1)
    with open(TUNNEL_FILE) as f:
        return json.load(f)


def patch_env(host: str, port: str) -> None:
    text = ENV_FILE.read_text(encoding="utf-8")
    text = re.sub(r"^MILVUS_HOST=.*", f"MILVUS_HOST={host}", text, flags=re.MULTILINE)
    text = re.sub(r"^MILVUS_PORT=.*", f"MILVUS_PORT={port}", text, flags=re.MULTILINE)
    # Disable Milvus Lite so the external tunnel is used
    text = re.sub(r"^USE_MILVUS_LITE=.*", "USE_MILVUS_LITE=false", text, flags=re.MULTILINE)
    text = re.sub(r"^MILVUS_LITE=.*", "MILVUS_LITE=false", text, flags=re.MULTILINE)
    ENV_FILE.write_text(text, encoding="utf-8")


def reset_env() -> None:
    """Restore .env to use localhost Milvus Lite (after Colab session ends)."""
    text = ENV_FILE.read_text(encoding="utf-8")
    text = re.sub(r"^MILVUS_HOST=.*", "MILVUS_HOST=localhost", text, flags=re.MULTILINE)
    text = re.sub(r"^MILVUS_PORT=.*", "MILVUS_PORT=19530", text, flags=re.MULTILINE)
    text = re.sub(r"^USE_MILVUS_LITE=.*", "USE_MILVUS_LITE=true", text, flags=re.MULTILINE)
    text = re.sub(r"^MILVUS_LITE=.*", "MILVUS_LITE=true", text, flags=re.MULTILINE)
    ENV_FILE.write_text(text, encoding="utf-8")
    print(".env restored to localhost / Milvus Lite mode.")


def verify_connection(host: str, port: str) -> bool:
    try:
        from pymilvus import connections, utility
        connections.connect(alias="colab_check", host=host, port=int(port), timeout=8)
        cols = utility.list_collections(using="colab_check")
        connections.disconnect("colab_check")
        print(f"Connection OK — collections on Colab Milvus: {cols or '(none yet)'}")
        return True
    except Exception as exc:
        print(f"[WARN] Connection check failed: {exc}")
        print("The tunnel may still be starting — try again in a few seconds.")
        return False


def main() -> None:
    if "--reset" in sys.argv:
        reset_env()
        return

    info = read_tunnel()
    host, port = info["host"], info["port"]

    patch_env(host, port)
    print(f".env updated  ->  MILVUS_HOST={host}  MILVUS_PORT={port}")

    verify_connection(host, port)

    if "--restart" in sys.argv:
        print("Restarting backend...")
        subprocess.Popen([sys.executable, "main.py"])
        print("Backend restarting at http://localhost:8000")


if __name__ == "__main__":
    main()
