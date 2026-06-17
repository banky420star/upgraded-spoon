"""
Backup Manager for Chain Gambler

Handles automated backup and recovery of critical trading system data:
- Model registry (active.json, candidates, champion, canary)
- Configuration files (config.yaml, per-symbol configs)
- Trading history and logs
- Model weights and checkpoints

Usage:
    from Python.backup_manager import BackupManager, get_backup_manager

    # Create backup manager
    backup_mgr = BackupManager()

    # Manual backup
    backup_path = backup_mgr.create_backup()

    # List available backups
    backups = backup_mgr.list_backups()

    # Restore from backup
    backup_mgr.restore_backup(backups[0]["path"])

    # Setup automated backups (every N hours)
    backup_mgr.start_automated_backups(interval_hours=24, max_backups=7)

Environment Variables:
    AGI_BACKUP_DIR: Directory for backups (default: project_root/backups)
    AGI_BACKUP_INTERVAL_HOURS: Automated backup interval (default: 24)
    AGI_MAX_BACKUPS: Maximum backups to keep (default: 7)
"""
import json
import os
import shutil
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger


class BackupManager:
    """Manage backups of critical trading system state."""

    # Paths relative to project root to backup
    BACKUP_PATHS = [
        "models/registry",           # Model registry state
        "config.yaml",               # Main configuration
        "configs",                   # Per-symbol configurations
        "logs/decisions.jsonl",      # Decision history
        "logs/monitoring",           # Monitoring data
    ]

    def __init__(self, project_root: Path = None, backup_dir: Path = None):
        """
        Initialize backup manager.

        Args:
            project_root: Project root directory (default: auto-detect)
            backup_dir: Directory for backups (default: project_root/backups)
        """
        if project_root is None:
            self.project_root = Path(__file__).resolve().parents[1]
        else:
            self.project_root = Path(project_root)

        if backup_dir is None:
            backup_dir = os.environ.get("AGI_BACKUP_DIR")
            if backup_dir:
                self.backup_dir = Path(backup_dir)
            else:
                self.backup_dir = self.project_root / "backups"
        else:
            self.backup_dir = Path(backup_dir)

        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Automated backup state
        self._automated_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._max_backups = int(os.environ.get("AGI_MAX_BACKUPS", "7"))

        logger.info(f"BackupManager initialized: project={self.project_root}, backups={self.backup_dir}")

    def create_backup(self, name: str = None, include_models: bool = False) -> Path:
        """
        Create a backup archive of critical system state.

        Args:
            name: Optional backup name (default: auto-generated timestamp)
            include_models: Whether to include full model weights (large!)

        Returns:
            Path to created backup archive
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = name or f"chain_gambler_backup_{timestamp}"
        backup_path = self.backup_dir / f"{name}.tar.gz"

        logger.info(f"Creating backup: {backup_path.name}")

        backed_up_files = []
        errors = []

        with tarfile.open(backup_path, "w:gz") as tar:
            for rel_path in self.BACKUP_PATHS:
                full_path = self.project_root / rel_path
                if full_path.exists():
                    try:
                        arcname = f"backup/{rel_path}"
                        tar.add(full_path, arcname=arcname)
                        backed_up_files.append(rel_path)
                        logger.debug(f"  Added: {rel_path}")
                    except Exception as e:
                        errors.append((rel_path, str(e)))
                        logger.warning(f"  Failed to add {rel_path}: {e}")
                else:
                    logger.debug(f"  Skipped (not found): {rel_path}")

            # Optionally include model weights
            if include_models:
                models_path = self.project_root / "models"
                if models_path.exists():
                    try:
                        # Add champion and canary models
                        for model_type in ["champion", "canary"]:
                            model_dir = models_path / "registry" / model_type
                            if model_dir.exists():
                                tar.add(model_dir, arcname=f"backup/models/{model_type}")
                                backed_up_files.append(f"models/{model_type}")
                        logger.info("  Included model weights (large)")
                    except Exception as e:
                        errors.append(("models", str(e)))
                        logger.warning(f"  Failed to add models: {e}")

            # Add backup metadata
            metadata = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "version": "1.0",
                "backed_up_files": backed_up_files,
                "errors": errors,
                "include_models": include_models,
            }
            metadata_bytes = json.dumps(metadata, indent=2).encode("utf-8")

            # Create a temporary file for metadata
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                f.write(json.dumps(metadata, indent=2))
                metadata_path = f.name

            try:
                tar.add(metadata_path, arcname="backup/metadata.json")
            finally:
                os.unlink(metadata_path)

        logger.success(f"Backup created: {backup_path} ({len(backed_up_files)} items, {len(errors)} errors)")

        # Prune old backups
        self._prune_old_backups()

        return backup_path

    def list_backups(self) -> List[Dict]:
        """
        List available backups with metadata.

        Returns:
            List of dicts with keys: path, name, created_at, size_bytes
        """
        backups = []

        if not self.backup_dir.exists():
            return backups

        for backup_file in sorted(self.backup_dir.glob("*.tar.gz"), reverse=True):
            try:
                stat = backup_file.stat()
                backups.append({
                    "path": backup_file,
                    "name": backup_file.stem,
                    "created_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                })
            except Exception as e:
                logger.warning(f"Failed to stat backup {backup_file}: {e}")

        return backups

    def restore_backup(self, backup_path: Path, dry_run: bool = False) -> Dict:
        """
        Restore system state from a backup archive.

        Args:
            backup_path: Path to backup archive
            dry_run: If True, only show what would be restored without making changes

        Returns:
            Dict with restoration results
        """
        backup_path = Path(backup_path)
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")

        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Restoring from: {backup_path.name}")

        restored = []
        errors = []

        with tarfile.open(backup_path, "r:gz") as tar:
            # Read metadata if available
            metadata = None
            try:
                metadata_file = tar.extractfile("backup/metadata.json")
                if metadata_file:
                    metadata = json.loads(metadata_file.read().decode("utf-8"))
                    logger.info(f"Backup created: {metadata.get('created_at', 'unknown')}")
            except Exception as e:
                logger.warning(f"Failed to read metadata: {e}")

            # List all items in backup
            for member in tar.getmembers():
                if member.isfile() or member.isdir():
                    # Extract relative path from backup/<path>
                    parts = member.name.split("/", 1)
                    if len(parts) == 2 and parts[0] == "backup":
                        rel_path = parts[1]
                        target_path = self.project_root / rel_path

                        if dry_run:
                            restored.append(rel_path)
                            continue

                        try:
                            # Ensure parent directory exists
                            if member.isfile():
                                target_path.parent.mkdir(parents=True, exist_ok=True)

                            # Extract file
                            tar.extract(member, path=self.project_root.parent)

                            # The extraction puts files at backup/<path>, we need to move them
                            extracted_path = self.project_root.parent / member.name
                            if extracted_path.exists():
                                if target_path.exists() and target_path.is_file():
                                    # Backup existing file
                                    backup_existing = target_path.with_suffix(f".backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}{target_path.suffix}")
                                    shutil.copy2(target_path, backup_existing)

                                # Move from extracted location to final location
                                if extracted_path.is_file():
                                    shutil.move(str(extracted_path), str(target_path))
                                elif extracted_path.is_dir():
                                    if target_path.exists():
                                        shutil.rmtree(target_path)
                                    shutil.move(str(extracted_path), str(target_path))

                                restored.append(rel_path)
                                logger.debug(f"  Restored: {rel_path}")
                        except Exception as e:
                            errors.append((rel_path, str(e)))
                            logger.warning(f"  Failed to restore {rel_path}: {e}")

        # Clean up backup directory structure if it was created
        backup_root = self.project_root.parent / "backup"
        if backup_root.exists() and not dry_run:
            try:
                shutil.rmtree(backup_root)
            except:
                pass

        result = {
            "restored_count": len(restored),
            "error_count": len(errors),
            "restored_items": restored,
            "errors": errors,
            "dry_run": dry_run,
        }

        if dry_run:
            logger.info(f"[DRY RUN] Would restore {len(restored)} items")
        else:
            logger.success(f"Restored {len(restored)} items ({len(errors)} errors)")

        return result

    def _prune_old_backups(self):
        """Remove old backups to keep only max_backups most recent."""
        backups = self.list_backups()

        if len(backups) > self._max_backups:
            to_delete = backups[self._max_backups:]
            for backup in to_delete:
                try:
                    backup["path"].unlink()
                    logger.debug(f"Pruned old backup: {backup['name']}")
                except Exception as e:
                    logger.warning(f"Failed to prune backup {backup['path']}: {e}")

            logger.info(f"Pruned {len(to_delete)} old backups")

    def start_automated_backups(self, interval_hours: int = None, max_backups: int = None):
        """
        Start automated backup thread.

        Args:
            interval_hours: Backup interval in hours (default: from env or 24)
            max_backups: Maximum backups to keep (default: from env or 7)
        """
        if interval_hours is None:
            interval_hours = int(os.environ.get("AGI_BACKUP_INTERVAL_HOURS", "24"))

        if max_backups is not None:
            self._max_backups = max_backups

        if self._automated_thread is not None and self._automated_thread.is_alive():
            logger.warning("Automated backups already running")
            return

        self._stop_event.clear()

        def backup_loop():
            logger.info(f"Automated backups started: interval={interval_hours}h, max_backups={self._max_backups}")

            while not self._stop_event.is_set():
                try:
                    self.create_backup()
                except Exception as e:
                    logger.error(f"Automated backup failed: {e}")

                # Wait for interval or until stopped
                for _ in range(interval_hours * 3600):
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)

        self._automated_thread = threading.Thread(target=backup_loop, daemon=True)
        self._automated_thread.start()

    def stop_automated_backups(self):
        """Stop the automated backup thread."""
        if self._automated_thread is None or not self._automated_thread.is_alive():
            logger.warning("Automated backups not running")
            return

        logger.info("Stopping automated backups...")
        self._stop_event.set()
        self._automated_thread.join(timeout=5)

        if self._automated_thread.is_alive():
            logger.warning("Backup thread did not stop gracefully")
        else:
            logger.info("Automated backups stopped")


# Global instance
_backup_manager: Optional[BackupManager] = None


def get_backup_manager() -> BackupManager:
    """Get or create global backup manager."""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager()
    return _backup_manager


if __name__ == "__main__":
    # CLI usage
    import argparse

    parser = argparse.ArgumentParser(description="Chain Gambler Backup Manager")
    parser.add_argument("command", choices=["create", "list", "restore", "auto"], help="Command to run")
    parser.add_argument("--include-models", action="store_true", help="Include model weights in backup")
    parser.add_argument("--backup-path", type=str, help="Backup path for restore command")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")

    args = parser.parse_args()

    mgr = BackupManager()

    if args.command == "create":
        backup = mgr.create_backup(include_models=args.include_models)
        print(f"Created backup: {backup}")

    elif args.command == "list":
        backups = mgr.list_backups()
        if not backups:
            print("No backups found")
        else:
            print(f"{'Name':<40} {'Created':<20} {'Size (MB)':<10}")
            print("-" * 70)
            for b in backups:
                print(f"{b['name']:<40} {b['created_at']:<20} {b['size_mb']:<10}")

    elif args.command == "restore":
        if not args.backup_path:
            print("Error: --backup-path required for restore")
            exit(1)

        result = mgr.restore_backup(Path(args.backup_path), dry_run=args.dry_run)
        print(f"Restored {result['restored_count']} items ({result['error_count']} errors)")
        if result['errors']:
            for path, error in result['errors']:
                print(f"  Error: {path} - {error}")

    elif args.command == "auto":
        mgr.start_automated_backups()
        print("Automated backups started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            mgr.stop_automated_backups()
