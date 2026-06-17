"""Stub Telegram alerter - forwards to alerts.telegram_alerts."""
import sys
import os

# Add parent to path for importing from alerts
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from alerts.telegram_alerts import TelegramAlerter

__all__ = ["TelegramAlerter"]
