import os


class MetaAPIExecutor:
    """
    Optional execution scaffold for future non-local MT5 routing.
    It intentionally mirrors the MT5Executor surface used by HybridBrain.
    """

    def __init__(self, risk):
        self.risk = risk
        self.token = os.environ.get("METAAPI_TOKEN", "")
        self.account_id = os.environ.get("METAAPI_ACCOUNT_ID", "")

    def get_tick(self, symbol):
        return None

    def reconcile_exposure(self, symbol, target_exposure, max_lots):
        raise RuntimeError("MetaAPIExecutor is a scaffold. Configure and wire a live MetaAPI session before use.")

    def manage_open_positions(self, symbol):
        return None
