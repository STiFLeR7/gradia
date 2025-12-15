from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np

class GradiaModel(ABC):
    """Abstract base class for all Gradia models."""

    @abstractmethod
    def fit(self, X, y, **kwargs):
        """Train the model fully."""
        pass

    def partial_fit(self, X, y, **kwargs):
        """Train on a batch or single epoch (optional)."""
        raise NotImplementedError("This model does not support iterative training.")

    @property
    def supports_iterative(self) -> bool:
        return False

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X) -> Optional[np.ndarray]:
        """Make probability predictions (if applicable)."""
        pass

    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return feature importance map if available."""
        pass
        
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return model hyperparameters."""
        pass
