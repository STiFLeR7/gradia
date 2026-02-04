"""
Timeline Logger for Learning Timeline (v2.0.0)

Handles persistent storage of learning events, compatible with existing
EventLogger infrastructure while extending for timeline data.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import time
import threading
import os

from .models import LearningEvent, EpochSummary


class TimelineLogger:
    """
    Logs learning events to structured files for timeline visualization.
    
    Storage format:
    - timeline_events.jsonl: Raw LearningEvents (append-only)
    - timeline_summary.jsonl: EpochSummaries
    - timeline_state.json: Tracker state for resumption
    
    Thread-safe via shared lock with existing EventLogger.
    """
    
    def __init__(self, log_dir: str, lock: Optional[threading.Lock] = None):
        """
        Initialize timeline logger.
        
        Args:
            log_dir: Directory for log files
            lock: Optional shared lock (uses global if not provided)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Use shared lock from callbacks module for thread safety
        if lock is None:
            from ..trainer.callbacks import log_lock
            self._lock = log_lock
        else:
            self._lock = lock
        
        # File paths
        self.events_path = self.log_dir / "timeline_events.jsonl"
        self.summary_path = self.log_dir / "timeline_summary.jsonl"
        self.state_path = self.log_dir / "timeline_state.json"
        
        # In-memory buffer for batch writes
        self._event_buffer: List[LearningEvent] = []
        self._buffer_size = 50  # Flush every N events
    
    def log_events(self, events: List[LearningEvent], flush: bool = False):
        """
        Log a batch of learning events.
        
        Args:
            events: List of LearningEvents to log
            flush: Force immediate write to disk
        """
        self._event_buffer.extend(events)
        
        if flush or len(self._event_buffer) >= self._buffer_size:
            self._flush_events()
    
    def log_event(self, event: LearningEvent):
        """Log a single learning event."""
        self.log_events([event])
    
    def log_summary(self, summary: EpochSummary):
        """Log an epoch summary."""
        with self._lock:
            with open(self.summary_path, "a") as f:
                f.write(json.dumps(summary.to_dict()) + "\n")
                f.flush()
                os.fsync(f.fileno())
    
    def save_tracker_state(self, tracker_data: Dict[str, Any]):
        """
        Save tracker state for run resumption/replay.
        
        Args:
            tracker_data: Serialized SampleTracker state
        """
        with self._lock:
            with open(self.state_path, "w") as f:
                json.dump(tracker_data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
    
    def load_tracker_state(self) -> Optional[Dict[str, Any]]:
        """Load saved tracker state if exists."""
        if not self.state_path.exists():
            return None
        
        with self._lock:
            with open(self.state_path, "r") as f:
                return json.load(f)
    
    def get_events(
        self,
        epoch: Optional[int] = None,
        sample_id: Optional[int] = None
    ) -> List[LearningEvent]:
        """
        Read events from log file with optional filtering.
        
        Args:
            epoch: Filter by epoch number
            sample_id: Filter by sample ID
        """
        events = []
        
        if not self.events_path.exists():
            return events
        
        with self._lock:
            with open(self.events_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        event = LearningEvent.from_dict(data)
                        
                        # Apply filters
                        if epoch is not None and event.epoch != epoch:
                            continue
                        if sample_id is not None and event.sample_id != sample_id:
                            continue
                        
                        events.append(event)
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        return events
    
    def get_summaries(self) -> List[EpochSummary]:
        """Read all epoch summaries."""
        summaries = []
        
        if not self.summary_path.exists():
            return summaries
        
        with self._lock:
            with open(self.summary_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        summary = EpochSummary(**data)
                        summaries.append(summary)
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
        
        return summaries
    
    def get_sample_timeline(self, sample_id: int) -> List[LearningEvent]:
        """Get full timeline for a specific sample."""
        return self.get_events(sample_id=sample_id)
    
    def get_latest_epoch(self) -> int:
        """Get the most recent epoch number logged."""
        summaries = self.get_summaries()
        if not summaries:
            return 0
        return max(s.epoch for s in summaries)
    
    def clear(self):
        """Clear all timeline logs (for new run)."""
        self._flush_events()  # Flush buffer first
        
        with self._lock:
            for path in [self.events_path, self.summary_path, self.state_path]:
                if path.exists():
                    path.unlink()
    
    def _flush_events(self):
        """Write buffered events to disk."""
        if not self._event_buffer:
            return
        
        with self._lock:
            with open(self.events_path, "a") as f:
                for event in self._event_buffer:
                    f.write(json.dumps(event.to_dict(), default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
        
        self._event_buffer.clear()
    
    def finalize(self):
        """Ensure all buffered data is written."""
        self._flush_events()
    
    def __del__(self):
        """Flush on deletion."""
        try:
            self._flush_events()
        except Exception:
            pass


def create_timeline_logger(run_dir: str) -> TimelineLogger:
    """
    Factory function to create a TimelineLogger.
    
    Convenience function that handles path resolution.
    """
    return TimelineLogger(run_dir)
