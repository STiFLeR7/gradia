"""
Sample Tracker for Learning Timeline (v2.0.0)

Manages which samples to track and maintains their state across epochs.
Implements deterministic, bounded sampling strategy.
"""

from typing import List, Dict, Any, Optional, Set
import numpy as np
from dataclasses import dataclass, field

from .models import LearningEvent, SampleState, EpochSummary


@dataclass
class SampleTracker:
    """
    Tracks a bounded subset of samples throughout training.
    
    Strategy:
    - Auto-select hard samples (near decision boundary)
    - Include user-selected samples if specified
    - Cap at max_samples for performance
    - Use deterministic seeding for reproducibility
    
    Attributes:
        max_samples: Maximum samples to track (default 100)
        seed: Random seed for reproducibility
        tracked_indices: Set of sample indices being tracked
        sample_states: State history for each tracked sample
    """
    max_samples: int = 100
    seed: int = 42
    tracked_indices: Set[int] = field(default_factory=set)
    sample_states: Dict[int, SampleState] = field(default_factory=dict)
    user_selected: Set[int] = field(default_factory=set)
    run_id: str = ""
    _initialized: bool = False
    
    def initialize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        run_id: str,
        user_indices: Optional[List[int]] = None,
        model: Optional[Any] = None
    ):
        """
        Initialize tracking with dataset and optional model predictions.
        
        Args:
            X: Feature matrix
            y: Labels
            run_id: Unique run identifier
            user_indices: User-selected sample indices to always track
            model: Optional model for boundary sample selection
        """
        self.run_id = run_id
        self._rng = np.random.RandomState(self.seed)
        
        n_samples = len(y)
        
        # Start with user-selected samples
        if user_indices:
            self.user_selected = set(user_indices[:self.max_samples // 2])
            self.tracked_indices = self.user_selected.copy()
        
        remaining_slots = self.max_samples - len(self.tracked_indices)
        
        if remaining_slots > 0:
            # Try to select boundary/hard samples if model available
            if model is not None and hasattr(model, 'predict_proba'):
                boundary_indices = self._select_boundary_samples(X, y, model, remaining_slots)
                self.tracked_indices.update(boundary_indices)
            
            # Fill remaining with stratified random
            remaining_slots = self.max_samples - len(self.tracked_indices)
            if remaining_slots > 0:
                available = set(range(n_samples)) - self.tracked_indices
                random_indices = self._stratified_sample(
                    list(available), y, remaining_slots
                )
                self.tracked_indices.update(random_indices)
        
        # Initialize sample states
        for idx in self.tracked_indices:
            self.sample_states[idx] = SampleState(
                sample_id=idx,
                true_label=y[idx] if hasattr(y, '__getitem__') else y.iloc[idx]
            )
        
        self._initialized = True
        
    def _select_boundary_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
        n_samples: int
    ) -> Set[int]:
        """
        Select samples near the decision boundary.
        
        These are the most informative for understanding model learning.
        """
        try:
            probas = model.predict_proba(X)
            
            # Compute margin: difference between top 2 class probabilities
            if probas.shape[1] >= 2:
                sorted_probas = np.sort(probas, axis=1)
                margins = sorted_probas[:, -1] - sorted_probas[:, -2]
            else:
                margins = np.abs(probas[:, 0] - 0.5)
            
            # Lower margin = closer to boundary = more interesting
            # Exclude already tracked
            available_mask = np.ones(len(margins), dtype=bool)
            for idx in self.tracked_indices:
                available_mask[idx] = False
            
            margins[~available_mask] = np.inf
            
            # Select lowest margin samples
            boundary_indices = np.argsort(margins)[:n_samples]
            return set(boundary_indices.tolist())
            
        except Exception:
            # Fallback if predict_proba fails
            return set()
    
    def _stratified_sample(
        self,
        available: List[int],
        y: np.ndarray,
        n_samples: int
    ) -> Set[int]:
        """
        Stratified random sampling to maintain class balance.
        """
        if not available:
            return set()
            
        # Group by class
        class_indices: Dict[Any, List[int]] = {}
        for idx in available:
            label = y[idx] if hasattr(y, '__getitem__') else y.iloc[idx]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # Sample proportionally from each class
        selected = []
        n_classes = len(class_indices)
        per_class = max(1, n_samples // n_classes)
        
        for label, indices in class_indices.items():
            k = min(per_class, len(indices))
            sampled = self._rng.choice(indices, size=k, replace=False)
            selected.extend(sampled.tolist())
        
        # Trim if over budget
        if len(selected) > n_samples:
            selected = self._rng.choice(selected, size=n_samples, replace=False).tolist()
            
        return set(selected)
    
    def record_predictions(
        self,
        epoch: int,
        X: np.ndarray,
        y: np.ndarray,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> List[LearningEvent]:
        """
        Record predictions for all tracked samples at this epoch.
        
        Args:
            epoch: Current epoch number
            X: Full feature matrix
            y: Full labels
            predictions: Model predictions for all samples
            probabilities: Optional probability matrix
            
        Returns:
            List of LearningEvents for this epoch
        """
        if not self._initialized:
            raise RuntimeError("SampleTracker not initialized. Call initialize() first.")
        
        events = []
        
        for idx in self.tracked_indices:
            true_label = y[idx] if hasattr(y, '__getitem__') else y.iloc[idx]
            pred_label = predictions[idx]
            
            # Compute confidence
            if probabilities is not None:
                proba_row = probabilities[idx]
                confidence = float(np.max(proba_row))
                proba_list = proba_row.tolist()
                
                # Compute margin
                sorted_p = np.sort(proba_row)
                margin = float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p) >= 2 else confidence
            else:
                confidence = 1.0  # No probability info
                proba_list = None
                margin = None
            
            correct = (pred_label == true_label)
            
            event = LearningEvent(
                run_id=self.run_id,
                epoch=epoch,
                sample_id=idx,
                true_label=true_label,
                predicted_label=pred_label,
                confidence=confidence,
                correct=bool(correct),
                margin=margin,
                probabilities=proba_list
            )
            
            # Update sample state
            self.sample_states[idx].add_event(event)
            events.append(event)
        
        return events
    
    def get_epoch_summary(self, epoch: int) -> EpochSummary:
        """
        Generate aggregated summary for an epoch.
        """
        states = list(self.sample_states.values())
        
        # Count by stability class
        stability_counts = {
            "stable_correct": 0,
            "stable_wrong": 0,
            "unstable": 0,
            "late_learner": 0,
            "unknown": 0
        }
        
        correct_count = 0
        flip_count = 0
        
        for state in states:
            stability_counts[state.stability_class] += 1
            if state.history and state.history[-1].correct:
                correct_count += 1
            flip_count += state.flip_count
        
        return EpochSummary(
            run_id=self.run_id,
            epoch=epoch,
            timestamp=__import__('time').time(),
            total_tracked=len(states),
            correct_count=correct_count,
            flip_count=flip_count,
            stable_correct=stability_counts["stable_correct"],
            stable_wrong=stability_counts["stable_wrong"],
            unstable=stability_counts["unstable"],
            late_learners=stability_counts["late_learner"]
        )
    
    def get_top_flipping_samples(self, n: int = 10) -> List[SampleState]:
        """Get samples with most prediction flips."""
        sorted_states = sorted(
            self.sample_states.values(),
            key=lambda s: s.flip_count,
            reverse=True
        )
        return sorted_states[:n]
    
    def get_late_learners(self, threshold_epoch: int = 5) -> List[SampleState]:
        """Get samples that became correct after threshold epoch."""
        late = []
        for state in self.sample_states.values():
            first = state.first_correct_epoch
            if first is not None and first >= threshold_epoch:
                late.append(state)
        return sorted(late, key=lambda s: s.first_correct_epoch or 999)
    
    def get_never_correct(self) -> List[SampleState]:
        """Get samples that were never correctly classified."""
        return [
            state for state in self.sample_states.values()
            if state.first_correct_epoch is None and state.history
        ]
    
    def get_sample_state(self, sample_id: int) -> Optional[SampleState]:
        """Get state for a specific sample."""
        return self.sample_states.get(sample_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize tracker state for storage."""
        return {
            "max_samples": self.max_samples,
            "seed": self.seed,
            "run_id": self.run_id,
            "tracked_indices": list(self.tracked_indices),
            "user_selected": list(self.user_selected),
            "sample_states": {
                idx: {
                    "sample_id": state.sample_id,
                    "true_label": state.true_label,
                    "history": [e.to_dict() for e in state.history]
                }
                for idx, state in self.sample_states.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleTracker":
        """Restore tracker from serialized state."""
        tracker = cls(
            max_samples=data["max_samples"],
            seed=data["seed"],
            run_id=data["run_id"]
        )
        tracker.tracked_indices = set(data["tracked_indices"])
        tracker.user_selected = set(data.get("user_selected", []))
        
        for idx_str, state_data in data["sample_states"].items():
            idx = int(idx_str)
            state = SampleState(
                sample_id=state_data["sample_id"],
                true_label=state_data["true_label"],
                history=[LearningEvent.from_dict(e) for e in state_data["history"]]
            )
            tracker.sample_states[idx] = state
        
        tracker._initialized = True
        return tracker
