"""
Gradia - Local-first ML Training Visualization

v2.0.0: Learning Timeline Edition
"""

__version__ = "2.0.0"

# Core exports
from .core.scenario import Scenario, ScenarioInferrer
from .core.config import ConfigManager
from .core.inspector import Inspector
from .core.migration import SchemaMigrator, ensure_v2_config

# Events module (v2.0)
from .events import LearningEvent, SampleTracker, TimelineLogger

# Trainer
from .trainer.engine import Trainer
from .trainer.callbacks import EventLogger

__all__ = [
    "__version__",
    # Core
    "Scenario",
    "ScenarioInferrer", 
    "ConfigManager",
    "Inspector",
    "SchemaMigrator",
    "ensure_v2_config",
    # Events (v2.0)
    "LearningEvent",
    "SampleTracker",
    "TimelineLogger",
    # Trainer
    "Trainer",
    "EventLogger",
]