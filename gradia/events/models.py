"""
Event Models for Learning Timeline (v2.0.0)

Defines the LearningEvent contract that all timeline visuals consume.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict, List
from enum import Enum
import time


class EventType(str, Enum):
    """Types of events in the learning timeline."""
    SAMPLE_PREDICTION = "sample_prediction"
    EPOCH_SUMMARY = "epoch_summary"
    FLIP_DETECTED = "flip_detected"
    STABILITY_CHANGE = "stability_change"


@dataclass
class LearningEvent:
    """
    Core event model for sample-level prediction tracking.
    
    This is the internal contract between training logic and visualization.
    All timeline visuals consume LearningEvents, not training internals.
    
    Attributes:
        run_id: Unique identifier for the training run
        epoch: Current epoch number (1-indexed)
        sample_id: Index or identifier of the tracked sample
        true_label: Ground truth label for the sample
        predicted_label: Model's prediction for this sample at this epoch
        confidence: Prediction confidence/probability (0.0 to 1.0)
        correct: Whether prediction matches true label
        timestamp: Unix timestamp when event was recorded
        margin: Optional decision margin (distance from decision boundary)
        probabilities: Optional full probability distribution across classes
        metadata: Optional additional context
    """
    run_id: str
    epoch: int
    sample_id: int
    true_label: Any
    predicted_label: Any
    confidence: float
    correct: bool
    timestamp: float = field(default_factory=time.time)
    margin: Optional[float] = None
    probabilities: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningEvent":
        """Reconstruct from dictionary."""
        return cls(**data)


@dataclass
class SampleState:
    """
    Tracks the learning state of a single sample across epochs.
    
    Used to compute stability metrics and detect flips.
    """
    sample_id: int
    true_label: Any
    history: List[LearningEvent] = field(default_factory=list)
    
    @property
    def flip_count(self) -> int:
        """Count how many times the prediction changed."""
        if len(self.history) < 2:
            return 0
        flips = 0
        for i in range(1, len(self.history)):
            if self.history[i].predicted_label != self.history[i-1].predicted_label:
                flips += 1
        return flips
    
    @property
    def is_stable(self) -> bool:
        """Sample is stable if no flips in last 3 epochs."""
        if len(self.history) < 3:
            return False
        last_3 = self.history[-3:]
        return all(e.predicted_label == last_3[0].predicted_label for e in last_3)
    
    @property
    def stability_class(self) -> str:
        """
        Classify sample stability for visualization.
        
        Returns:
            'stable_correct': Consistently correct
            'stable_wrong': Consistently wrong  
            'unstable': Predictions keep changing
            'late_learner': Recently became correct
        """
        if not self.history:
            return "unknown"
            
        recent = self.history[-3:] if len(self.history) >= 3 else self.history
        all_correct = all(e.correct for e in recent)
        all_wrong = all(not e.correct for e in recent)
        
        if all_correct and self.is_stable:
            return "stable_correct"
        elif all_wrong and self.is_stable:
            return "stable_wrong"
        elif self.flip_count > 2:
            return "unstable"
        elif len(self.history) >= 3 and self.history[-1].correct and not self.history[-3].correct:
            return "late_learner"
        else:
            return "unstable"
    
    @property 
    def first_correct_epoch(self) -> Optional[int]:
        """Return epoch when sample was first correctly classified."""
        for event in self.history:
            if event.correct:
                return event.epoch
        return None
    
    @property
    def current_prediction(self) -> Optional[Any]:
        """Most recent prediction."""
        return self.history[-1].predicted_label if self.history else None
    
    @property
    def current_confidence(self) -> Optional[float]:
        """Most recent confidence score."""
        return self.history[-1].confidence if self.history else None
    
    def add_event(self, event: LearningEvent):
        """Record a new prediction event."""
        self.history.append(event)


@dataclass  
class EpochSummary:
    """
    Aggregated summary of sample-level events for one epoch.
    
    Used for the Timeline Overview block.
    """
    run_id: str
    epoch: int
    timestamp: float
    total_tracked: int
    correct_count: int
    flip_count: int
    stable_correct: int
    stable_wrong: int
    unstable: int
    late_learners: int
    
    @property
    def accuracy(self) -> float:
        """Tracked sample accuracy."""
        return self.correct_count / self.total_tracked if self.total_tracked > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
