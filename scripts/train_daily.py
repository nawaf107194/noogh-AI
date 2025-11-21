#!/usr/bin/env python3
"""
🔁 Daily Training Automation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Automatic daily training pipeline:
- Fetch new data
- Update knowledge index
- Retrain models if needed
- Create backups
- Generate reports
"""

import os
import sys
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [TRAIN] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f"train_daily_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger(__name__)


class DailyTrainer:
    """Daily training automation system"""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = self.project_root / "data"
        self.backup_dir = self.project_root / "backups"
        self.models_dir = self.project_root / "models"

        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

        # Training stats
        self.stats = {
            "start_time": datetime.now().isoformat(),
            "tasks_completed": [],
            "tasks_failed": [],
            "data_updates": {},
            "model_updates": {}
        }

    def log_task(self, task_name: str, success: bool = True, details: str = ""):
        """Log task completion"""
        task_info = {
            "task": task_name,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }

        if success:
            self.stats["tasks_completed"].append(task_info)
            logger.info(f"✅ {task_name} completed. {details}")
        else:
            self.stats["tasks_failed"].append(task_info)
            logger.error(f"❌ {task_name} failed. {details}")

    def create_backup(self):
        """Create backup of current data and models"""
        logger.info("Creating daily backup...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"
        backup_path.mkdir(exist_ok=True)

        try:
            # Backup data
            if self.data_dir.exists():
                data_backup = backup_path / "data"
                shutil.copytree(self.data_dir, data_backup, dirs_exist_ok=True)
                logger.info(f"Data backed up to {data_backup}")

            # Backup brain memories
            brain_file = self.data_dir / "brain_v4_memories.jsonl"
            if brain_file.exists():
                shutil.copy(brain_file, backup_path / "brain_v4_memories.jsonl")

            # Backup knowledge index
            index_file = self.data_dir / "simple_index.json"
            if index_file.exists():
                shutil.copy(index_file, backup_path / "simple_index.json")

            # Clean old backups (keep last 7 days)
            self._clean_old_backups(days=7)

            self.log_task("Backup Creation", True, f"Saved to {backup_path}")
            return True

        except Exception as e:
            self.log_task("Backup Creation", False, str(e))
            return False

    def _clean_old_backups(self, days: int = 7):
        """Remove backups older than specified days"""
        if not self.backup_dir.exists():
            return

        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)

        for backup in self.backup_dir.iterdir():
            if backup.is_dir() and backup.stat().st_mtime < cutoff_time:
                shutil.rmtree(backup)
                logger.info(f"Removed old backup: {backup.name}")

    def fetch_new_data(self):
        """Fetch or generate new training data"""
        logger.info("Fetching new data...")

        try:
            # In production, this would fetch from external sources
            # For now, we'll simulate by checking for new files

            new_data_count = 0

            # Check for new data files
            data_sources = [
                self.data_dir / "training",
                self.data_dir / "reflection"
            ]

            for source in data_sources:
                if source.exists():
                    files = list(source.glob("*.json"))
                    new_data_count += len(files)

            self.stats["data_updates"]["new_files"] = new_data_count

            self.log_task("Data Fetching", True, f"Found {new_data_count} data files")
            return True

        except Exception as e:
            self.log_task("Data Fetching", False, str(e))
            return False

    def update_knowledge_index(self):
        """Update knowledge index with new information"""
        logger.info("Updating knowledge index...")

        try:
            # Run knowledge expansion script
            import subprocess

            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "expand_knowledge.py")],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                # Parse output to get chunk count
                output = result.stdout
                if "Total chunks:" in output:
                    chunks_line = [line for line in output.split('\n') if "Total chunks:" in line][0]
                    total_chunks = chunks_line.split(':')[1].strip()
                    self.stats["data_updates"]["total_chunks"] = total_chunks

                self.log_task("Knowledge Index Update", True, f"Index updated successfully")
                return True
            else:
                self.log_task("Knowledge Index Update", False, result.stderr)
                return False

        except Exception as e:
            self.log_task("Knowledge Index Update", False, str(e))
            return False

    def train_models(self):
        """Train or update models if needed"""
        logger.info("Checking if model training is needed...")

        try:
            # In production, this would check metrics and retrain if needed
            # For now, we'll just log the check

            # Check if we have enough new data
            new_files = self.stats["data_updates"].get("new_files", 0)

            if new_files > 10:  # Threshold for retraining
                logger.info(f"Sufficient new data ({new_files} files). Training recommended.")
                self.stats["model_updates"]["training_recommended"] = True
            else:
                logger.info(f"Insufficient new data ({new_files} files). Skipping training.")
                self.stats["model_updates"]["training_recommended"] = False

            self.log_task("Model Training Check", True, f"New files: {new_files}")
            return True

        except Exception as e:
            self.log_task("Model Training Check", False, str(e))
            return False

    def generate_report(self):
        """Generate training report"""
        logger.info("Generating training report...")

        try:
            self.stats["end_time"] = datetime.now().isoformat()

            # Calculate duration
            start = datetime.fromisoformat(self.stats["start_time"])
            end = datetime.fromisoformat(self.stats["end_time"])
            duration = (end - start).total_seconds()
            self.stats["duration_seconds"] = duration

            # Save report
            report_file = self.data_dir / "training" / f"daily_report_{datetime.now().strftime('%Y%m%d')}.json"
            report_file.parent.mkdir(exist_ok=True)

            with open(report_file, "w") as f:
                json.dump(self.stats, f, indent=2)

            # Generate summary
            summary = self._generate_summary()

            logger.info("\n" + "=" * 60)
            logger.info("📊 DAILY TRAINING SUMMARY")
            logger.info("=" * 60)
            logger.info(summary)
            logger.info("=" * 60)

            self.log_task("Report Generation", True, f"Saved to {report_file}")
            return True

        except Exception as e:
            self.log_task("Report Generation", False, str(e))
            return False

    def _generate_summary(self) -> str:
        """Generate human-readable summary"""
        lines = []
        lines.append(f"Start Time: {self.stats['start_time']}")
        lines.append(f"End Time: {self.stats['end_time']}")
        lines.append(f"Duration: {self.stats['duration_seconds']:.2f} seconds")
        lines.append(f"\nTasks Completed: {len(self.stats['tasks_completed'])}")
        lines.append(f"Tasks Failed: {len(self.stats['tasks_failed'])}")

        if self.stats['data_updates']:
            lines.append(f"\nData Updates:")
            for key, value in self.stats['data_updates'].items():
                lines.append(f"  - {key}: {value}")

        if self.stats['model_updates']:
            lines.append(f"\nModel Updates:")
            for key, value in self.stats['model_updates'].items():
                lines.append(f"  - {key}: {value}")

        return "\n".join(lines)

    def run(self):
        """Execute daily training pipeline"""
        logger.info("🚀 Starting Daily Training Pipeline")
        logger.info(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. Create backup
        self.create_backup()

        # 2. Fetch new data
        self.fetch_new_data()

        # 3. Update knowledge index
        self.update_knowledge_index()

        # 4. Check if training is needed
        self.train_models()

        # 5. Generate report
        self.generate_report()

        # Summary
        total_tasks = len(self.stats['tasks_completed']) + len(self.stats['tasks_failed'])
        success_rate = (len(self.stats['tasks_completed']) / total_tasks * 100) if total_tasks > 0 else 0

        logger.info(f"\n✅ Daily Training Pipeline Complete!")
        logger.info(f"   Success Rate: {success_rate:.1f}% ({len(self.stats['tasks_completed'])}/{total_tasks})")

        return len(self.stats['tasks_failed']) == 0


def main():
    """Main entry point"""
    try:
        trainer = DailyTrainer()
        success = trainer.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
