from loguru import logger

try:
    from Python.config_utils import get_symbol_config
except ImportError:
    get_symbol_config = None

# Import order placement matching Lane B pattern
try:
    from training.live_trade_lane_b import place_order
except ImportError:
    def place_order(symbol, direction, volume, sl_price=0.0, comment=""):
        logger.warning(f"Mock place_order: {direction} {volume} {symbol}")
        return True


class LiveTrader:
    """Sends real MT5 orders, gated by paper-mode approval."""

    def __init__(self, symbol: str, dry_run: bool = False):
        self.symbol = symbol
        self.dry_run = dry_run
        self.config = get_symbol_config(symbol) if get_symbol_config else {}
        # Risk gate
        if not self.config.get("risk", {}).get("live_approved", False):
            raise PermissionError(
                f"Live trading NOT approved for {self.symbol} in config!"
            )

    def execute_live_order(self, direction: str, current_price: float):
        volume = self.config.get("trading", {}).get("base_lot", 0.1)
        # TODO: Compute actual SL based on ATR or model confidence
        sl_price = (
            current_price * 0.99
            if direction == "BUY"
            else current_price * 1.01
        )
        if self.dry_run:
            logger.info(f"[DRY RUN] Would execute {direction} {volume} {self.symbol}")
            return
        success = place_order(
            self.symbol, direction, volume, sl_price, "SupremeChainsaw Live"
        )
        if success:
            logger.success(f"Placed {direction} for {self.symbol}")
        else:
            logger.error(f"Failed to place {direction} for {self.symbol}")


if __name__ == "__main__":
    trader = LiveTrader("XAUUSDm", dry_run=True)
