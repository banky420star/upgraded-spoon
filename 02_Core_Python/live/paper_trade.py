import time
from loguru import logger

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

try:
    from Python.config_utils import get_symbol_config
except ImportError:
    get_symbol_config = None


class PaperTrader:
    """Simulates trades without real MT5 execution."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.config = get_symbol_config(symbol) if get_symbol_config else {}
        # TODO: Load DRL model and feature pipeline

    def run_loop(self):
        logger.info(f"Starting Paper Trading loop for {self.symbol}")
        while True:
            try:
                self.simulate_tick()
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Paper trading stopped by user.")
                break

    def simulate_tick(self):
        if mt5 is None:
            logger.debug("MT5 not installed; acting on synthetic tick.")
        # TODO: Fetch real tick, predict via self.model
        action = "BUY"
        volume = self.config.get("trading", {}).get("base_lot", 0.1)
        price = 1950.00  # Synthetic fetched price
        self.log_decision(action, volume, price)

    def log_decision(self, action: str, volume: float, price: float):
        logger.info(f"[PAPER_TRADE] {action} {volume} {self.symbol} @ {price}")


if __name__ == "__main__":
    trader = PaperTrader("XAUUSDm")
    # trader.run_loop()  # Uncomment to run interactively
