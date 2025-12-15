import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigManager:
    """Manages gradia configuration."""
    
    DEFAULT_CONFIG = {
        'model': {
            'type': 'auto', # auto, linear, random_forest
            'params': {}
        },
        'training': {
            'test_split': 0.2,
            'random_seed': 42,
            'shuffle': True
        },
        'scenario': {
            'target': None, # Auto-detect
            'task': None # Auto-detect
        }
    }

    def __init__(self, run_dir: str = ".gradia_logs"):
        self.run_dir = Path(run_dir)
        self.config_path = self.run_dir / "config.yaml"

    def load_or_create(self, user_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        config = self.DEFAULT_CONFIG.copy()
        
        # Load existing if any (feature for restart, maybe not for MVP run-once)
        # For immutable runs, we usually generate NEW config.
        # But if gradia.yaml exists in ROOT, we load it.
        
        root_config = Path("gradia.yaml")
        if root_config.exists():
            with open(root_config, 'r') as f:
                user_config = yaml.safe_load(f)
                self._update_recursive(config, user_config)

        if user_overrides:
            self._update_recursive(config, user_overrides)
            
        return config

    def save(self, config: Dict[str, Any]):
        self.run_dir.mkdir(exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)

    def _update_recursive(self, base: Dict, update: Dict):
        for k, v in update.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._update_recursive(base[k], v)
            else:
                base[k] = v
