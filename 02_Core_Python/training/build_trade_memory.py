import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Python.trade_learning import build_trade_learning


def main():
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    out_dir = os.path.join(log_dir, "learning")
    lookback_days = int(os.environ.get("AGI_TRADE_LEARN_DAYS", "30"))
    summary = build_trade_learning(log_dir=log_dir, out_dir=out_dir, lookback_days=lookback_days)
    print(json.dumps({"ok": True, "summary": summary}, ensure_ascii=True))


if __name__ == "__main__":
    main()

