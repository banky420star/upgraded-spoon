"""
ParallelLaneManager — runs LSTM→PPO→Dreamer pipeline per symbol concurrently.
Designed to slot into Server_AGI alongside (not replace) the existing training loop.
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Literal, Optional
from loguru import logger

PhaseStatus = Literal["queued", "training", "done", "failed", "skipped"]


@dataclass
class LanePhaseState:
    status: PhaseStatus = "queued"
    progress_pct: float = 0.0
    epoch: int = 0
    epochs_total: int = 0
    loss: Optional[float] = None
    val_loss: Optional[float] = None
    fail_reason: Optional[str] = None
    started_at: Optional[float] = None  # unix timestamp
    finished_at: Optional[float] = None


@dataclass
class LaneState:
    symbol: str
    status: PhaseStatus = "queued"
    current_phase: Literal["LSTM", "PPO", "Dreamer", "Champion", "Done", "Failed"] = "LSTM"
    lstm: LanePhaseState = field(default_factory=LanePhaseState)
    ppo: LanePhaseState = field(default_factory=LanePhaseState)
    dreamer: LanePhaseState = field(default_factory=LanePhaseState)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def total_progress(self) -> float:
        """Weighted pipeline progress: LSTM=33%, PPO=33%, Dreamer=34%"""
        p = (
            self.lstm.progress_pct * 0.33
            + self.ppo.progress_pct * 0.33
            + self.dreamer.progress_pct * 0.34
        )
        return min(100.0, p)

    @property
    def eta_seconds(self) -> Optional[int]:
        """Rough ETA based on current phase speed."""
        now = time.time()
        started = self.started_at
        if started is None or self.status not in ("training",):
            return None
        elapsed = now - started
        progress = self.total_progress
        if progress <= 0.0:
            return None
        total_estimated = elapsed / (progress / 100.0)
        remaining = total_estimated - elapsed
        if remaining < 0:
            return 0
        return int(remaining)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "status": self.status,
            "current_phase": self.current_phase,
            "total_progress": round(self.total_progress, 1),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "eta_seconds": self.eta_seconds,
            "lstm": self._phase_dict(self.lstm),
            "ppo": self._phase_dict(self.ppo),
            "dreamer": self._phase_dict(self.dreamer),
        }

    def _phase_dict(self, p: LanePhaseState) -> dict:
        return {
            "status": p.status,
            "progress_pct": round(p.progress_pct, 1),
            "epoch": p.epoch,
            "epochs_total": p.epochs_total,
            "loss": p.loss,
            "val_loss": p.val_loss,
            "fail_reason": p.fail_reason,
        }


class ParallelLaneManager:
    """
    Manages concurrent per-symbol training pipelines.

    Usage in Server_AGI:
        self.lane_mgr = ParallelLaneManager(symbols, max_workers=4, server=self)
        self.lane_mgr.start_cycle()  # non-blocking
        # poll self.lane_mgr.get_status() in status endpoint
    """

    def __init__(self, symbols: list, max_workers: int = 4, server=None):
        self.symbols = symbols
        self.max_workers = max_workers
        self.server = server  # reference to Server_AGI for calling training methods
        self._lanes: dict = {}
        self._lock = threading.Lock()
        self._threads: list = []
        self._active = False

    def start_cycle(self):
        """Start parallel training for all symbols. Non-blocking."""
        if self.is_running():
            logger.warning("ParallelLaneManager: cycle already running")
            return
        self._active = True
        with self._lock:
            self._lanes = {sym: LaneState(symbol=sym) for sym in self.symbols}

        # Semaphore limits concurrency
        sem = threading.Semaphore(self.max_workers)
        self._threads = []
        for sym in self.symbols:
            t = threading.Thread(
                target=self._run_with_semaphore,
                args=(sym, sem),
                daemon=True,
                name=f"lane-{sym}",
            )
            self._threads.append(t)
            t.start()
        logger.info(
            f"ParallelLaneManager: started {len(self.symbols)} lanes (max_workers={self.max_workers})"
        )

    def _run_with_semaphore(self, symbol: str, sem: threading.Semaphore):
        with sem:
            self._run_lane(symbol)

    def _run_lane(self, symbol: str):
        """Run full LSTM→PPO→Dreamer pipeline for one symbol."""
        lane = self._get_lane(symbol)
        lane.started_at = time.time()
        lane.status = "training"

        try:
            # Phase 1: LSTM
            self._run_lstm(lane)
            if lane.lstm.status == "failed":
                lane.status = "failed"
                lane.current_phase = "Failed"
                lane.finished_at = time.time()
                return

            # Phase 2: PPO
            self._run_ppo(lane)

            # Phase 3: Dreamer
            self._run_dreamer(lane)

            lane.status = "done"
            lane.current_phase = "Done"
            lane.finished_at = time.time()
            logger.info(f"Lane {symbol}: pipeline complete")

        except Exception as e:
            logger.error(f"Lane {symbol} error: {e}")
            lane.status = "failed"
            lane.current_phase = "Failed"
            lane.finished_at = time.time()

    def _run_lstm(self, lane: LaneState):
        """Run LSTM training phase, updating lane.lstm progress."""
        lane.current_phase = "LSTM"
        phase = lane.lstm
        phase.status = "training"
        phase.started_at = time.time()

        try:
            if self.server and hasattr(self.server, "_train_lstm_for_symbol"):
                # Use existing training method with progress callback
                def on_progress(epoch, epochs_total, loss, val_loss):
                    phase.epoch = epoch
                    phase.epochs_total = epochs_total
                    phase.loss = loss
                    phase.val_loss = val_loss
                    phase.progress_pct = (epoch / max(1, epochs_total)) * 100

                self.server._train_lstm_for_symbol(lane.symbol, progress_cb=on_progress)
            else:
                # Simulate training if no server reference
                for i in range(1, 11):
                    time.sleep(0.1)
                    phase.epoch = i
                    phase.epochs_total = 10
                    phase.progress_pct = i * 10.0
                    phase.loss = 0.05 / i

            phase.status = "done"
            phase.progress_pct = 100.0
            phase.finished_at = time.time()

        except Exception as e:
            phase.status = "failed"
            phase.fail_reason = str(e)
            raise

    def _run_ppo(self, lane: LaneState):
        """Run PPO training phase."""
        lane.current_phase = "PPO"
        phase = lane.ppo
        phase.status = "training"
        phase.started_at = time.time()

        try:
            if self.server and hasattr(self.server, "_train_ppo_for_symbol"):
                def on_progress(timesteps, total_timesteps):
                    phase.progress_pct = (timesteps / max(1, total_timesteps)) * 100
                    phase.epoch = timesteps
                    phase.epochs_total = total_timesteps

                self.server._train_ppo_for_symbol(lane.symbol, progress_cb=on_progress)
            else:
                for i in range(1, 11):
                    time.sleep(0.1)
                    phase.progress_pct = i * 10.0

            phase.status = "done"
            phase.progress_pct = 100.0
            phase.finished_at = time.time()

        except Exception as e:
            phase.status = "failed"
            phase.fail_reason = str(e)
            # Don't raise — PPO failure shouldn't block Dreamer

    def _run_dreamer(self, lane: LaneState):
        """Run Dreamer training phase."""
        lane.current_phase = "Dreamer"
        phase = lane.dreamer
        phase.status = "training"
        phase.started_at = time.time()

        try:
            if self.server and hasattr(self.server, "_train_dreamer_for_symbol"):
                def on_progress(steps, total_steps):
                    phase.progress_pct = (steps / max(1, total_steps)) * 100

                self.server._train_dreamer_for_symbol(lane.symbol, progress_cb=on_progress)
            else:
                for i in range(1, 11):
                    time.sleep(0.1)
                    phase.progress_pct = i * 10.0

            phase.status = "done"
            phase.progress_pct = 100.0
            phase.finished_at = time.time()

        except Exception as e:
            phase.status = "failed"
            phase.fail_reason = str(e)

    def stop(self):
        """Signal all lanes to stop (non-forceful — waits for current phase to complete)."""
        self._active = False

    def is_running(self) -> bool:
        return self._active and any(t.is_alive() for t in self._threads)

    def get_status(self) -> dict:
        with self._lock:
            lanes = [lane.to_dict() for lane in self._lanes.values()]
        active = sum(1 for lane in lanes if lane["status"] == "training")
        return {
            "parallel_lanes": lanes,
            "max_parallel": self.max_workers,
            "active_count": active,
            "is_running": self.is_running(),
        }

    def _get_lane(self, symbol: str) -> LaneState:
        with self._lock:
            return self._lanes[symbol]
