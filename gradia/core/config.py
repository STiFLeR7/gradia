import yaml
from pathlib import Path
from typing import Any, Dict

from .migration import SchemaMigrator, SchemaVersion


class ConfigManager:
    """Manages gradia configuration with v2.0 migration support."""
    
    # v2.0 Default Configuration
    DEFAULT_CONFIG = {
        'schema_version': SchemaVersion.V2_0.value,
        'model': {
            'type': 'auto',  # auto, linear, random_forest, sgd, mlp, cnn
            'params': {}
        },
        'training': {
            'test_split': 0.2,
            'random_seed': 42,
            'shuffle': True,
            'epochs': 10
        },
        'scenario': {
            'target': None,  # Auto-detect
            'task': None  # Auto-detect
        },
        # v2.0: Learning Timeline configuration
        'timeline': {
            'enabled': True,
            'max_samples': 100,
            'user_samples': None  # List of sample indices to always track
        },
        'project_name': 'experiment',
        'save_model': False
    }

    def __init__(self, run_dir: str = ".gradia_logs"):
        self.run_dir = Path(run_dir)
        self.config_path = self.run_dir / "config.yaml"
        self._migrator = SchemaMigrator()

    def load_or_create(self, user_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        config = self._deep_copy(self.DEFAULT_CONFIG)
        
        # Load existing config if present (for run continuation)
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                existing = yaml.safe_load(f) or {}
                
                # Migrate to v2 if needed
                result = self._migrator.migrate(existing)
                if result.changes:
                    print(f"Config migrated: {', '.join(result.changes)}")
                
                self._update_recursive(config, existing)
        
        # Load root gradia.yaml overrides
        root_config = Path("gradia.yaml")
        if root_config.exists():
            with open(root_config, 'r') as f:
                user_config = yaml.safe_load(f) or {}
                self._update_recursive(config, user_config)

        # Apply explicit overrides
        if user_overrides:
            self._update_recursive(config, user_overrides)
        
        # Ensure v2 fields exist
        config = self._ensure_v2_fields(config)
            
        return config

    def save(self, config: Dict[str, Any]):
        """Save config with schema version marker."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure schema version is set
        config['schema_version'] = SchemaVersion.V2_0.value
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def _update_recursive(self, base: Dict, update: Dict):
        """Recursively merge update into base."""
        for k, v in update.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                self._update_recursive(base[k], v)
            else:
                base[k] = v
    
    def _deep_copy(self, d: Dict) -> Dict:
        """Create a deep copy of nested dict."""
        import copy
        return copy.deepcopy(d)
    
    def _ensure_v2_fields(self, config: Dict) -> Dict:
        """Ensure all v2.0 required fields exist."""
        # Timeline config
        if 'timeline' not in config:
            config['timeline'] = {
                'enabled': True,
                'max_samples': 100,
                'user_samples': None
            }
        
        # Training epochs
        if 'epochs' not in config.get('training', {}):
            config.setdefault('training', {})['epochs'] = 10
        
        # Schema version
        config['schema_version'] = SchemaVersion.V2_0.value
        
        return config
