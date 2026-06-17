"""
n8n ↔ AGI Bridge
Usage:
  python agi_n8n_bridge.py <COMMAND> <SYMBOL> [AGGRESSION]

COMMAND: predict | trade | health | risk_status
"""
import os
import sys
import socket
import json

HOST = os.environ.get("AGI_HOST", "127.0.0.1")
PORT = int(os.environ.get("AGI_PORT", "9090"))
TOKEN = os.environ.get("AGI_TOKEN", "").strip()

def _die(msg: dict, code: int = 1):
    print(json.dumps(msg))
    sys.exit(code)

def main():
    if len(sys.argv) < 3:
        _die({
            "error": "Usage: python agi_n8n_bridge.py <COMMAND> <SYMBOL> [AGGRESSION]",
            "action": "ERROR",
            "confidence": 0.0,
        })

    command = sys.argv[1].strip().lower()
    symbol = sys.argv[2].strip()
    aggression = (sys.argv[3].strip().lower() if len(sys.argv) >= 4 else "moderate")

    request = {
        "action": command,
        "symbol": symbol,
        "direction": "AUTO",
        "confidence": 0.0,
        "aggression": aggression,
    }
    if TOKEN:
        request["token"] = TOKEN

    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(10.0)
        client.connect((HOST, PORT))

        payload = (json.dumps(request) + "\n").encode("utf-8")
        client.sendall(payload)

        # read one line JSON response
        buf = b""
        while b"\n" not in buf:
            chunk = client.recv(4096)
            if not chunk:
                break
            buf += chunk

        client.close()
        raw = buf.decode("utf-8", errors="replace").strip()
        if not raw:
            _die({"error": "Empty response from server", "action": "ERROR", "symbol": symbol})

        try:
            result = json.loads(raw.splitlines()[0])
        except json.JSONDecodeError:
            result = {"error": "Failed to parse server JSON", "raw_response": raw, "action": "ERROR", "symbol": symbol}

        if "action" not in result:
            result["action"] = result.get("status", "UNKNOWN")

        print(json.dumps(result))

    except ConnectionRefusedError:
        _die({
            "error": f"AGI Server not running on {HOST}:{PORT}",
            "symbol": symbol,
            "confidence": 0.0,
            "action": "ERROR",
        })
    except Exception as e:
        _die({
            "error": str(e),
            "symbol": symbol,
            "confidence": 0.0,
            "action": "ERROR",
        })

if __name__ == "__main__":
    main()
