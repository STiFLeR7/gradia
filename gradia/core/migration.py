"""
Migration Layer for Gradia v2.0.0

Handles backward compatibility with v1.x .gradia_logs runs.
Supports schema versioning and silent migration.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import yaml
from dataclasses import dataclass
from enum import Enum


class SchemaVersion(str, Enum):
    """Gradia config/log schema versions."""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V1_2 = "1.2"
    V1_3 = "1.3"
    V2_0 = "2.0"
    
    @classmethod
    def current(cls) -> "SchemaVersion":
        return cls.V2_0


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    success: bool
    from_version: str
    to_version: str
    changes: List[str]
    warnings: List[str]


class SchemaMigrator:
    """
    Handles schema migration between Gradia versions.
    
    Strategy:
    - Silent upgrades (no user intervention required)
    - Deprecate, don't remove fields
    - Add new fields with sensible defaults
    """
    
    def __init__(self):
        self.migrations = {
            ("1.0", "1.1"): self._migrate_1_0_to_1_1,
            ("1.1", "1.2"): self._migrate_1_1_to_1_2,
            ("1.2", "1.3"): self._migrate_1_2_to_1_3,
            ("1.3", "2.0"): self._migrate_1_3_to_2_0,
        }
    
    def detect_version(self, config: Dict[str, Any]) -> str:
        """Detect schema version from config structure."""
        # v2.0 has explicit version field
        if "schema_version" in config:
            return config["schema_version"]
        
        # v2.0 has timeline config
        if "timeline" in config:
            return "2.0"
        
        # v1.3 has project_name and save_model
        if "project_name" in config or "save_model" in config:
            return "1.3"
        
        # v1.2 has training.epochs
        if config.get("training", {}).get("epochs"):
            return "1.2"
        
        # v1.1 has model.params
        if config.get("model", {}).get("params"):
            return "1.1"
        
        # Default to 1.0
        return "1.0"
    
    def migrate(self, config: Dict[str, Any], target_version: str = None) -> MigrationResult:
        """
        Migrate config to target version (default: current).
        
        Args:
            config: Configuration dictionary
            target_version: Target version string (default: current)
            
        Returns:
            MigrationResult with details
        """
        if target_version is None:
            target_version = SchemaVersion.current().value
        
        from_version = self.detect_version(config)
        changes = []
        warnings = []
        
        if from_version == target_version:
            return MigrationResult(
                success=True,
                from_version=from_version,
                to_version=target_version,
                changes=["No migration needed"],
                warnings=[]
            )
        
        # Build migration path
        current = from_version
        version_order = ["1.0", "1.1", "1.2", "1.3", "2.0"]
        
        try:
            start_idx = version_order.index(current)
            end_idx = version_order.index(target_version)
        except ValueError as e:
            return MigrationResult(
                success=False,
                from_version=from_version,
                to_version=target_version,
                changes=[],
                warnings=[f"Unknown version: {e}"]
            )
        
        if start_idx > end_idx:
            # Downgrade not supported
            return MigrationResult(
                success=False,
                from_version=from_version,
                to_version=target_version,
                changes=[],
                warnings=["Downgrade not supported. Please use a compatible Gradia version."]
            )
        
        # Apply migrations sequentially
        for i in range(start_idx, end_idx):
            v_from = version_order[i]
            v_to = version_order[i + 1]
            key = (v_from, v_to)
            
            if key in self.migrations:
                migration_fn = self.migrations[key]
                step_changes, step_warnings = migration_fn(config)
                changes.extend(step_changes)
                warnings.extend(step_warnings)
        
        # Set version marker
        config["schema_version"] = target_version
        
        return MigrationResult(
            success=True,
            from_version=from_version,
            to_version=target_version,
            changes=changes,
            warnings=warnings
        )
    
    def _migrate_1_0_to_1_1(self, config: Dict) -> tuple:
        """Add model.params if missing."""
        changes = []
        warnings = []
        
        if "model" not in config:
            config["model"] = {"type": "auto", "params": {}}
            changes.append("Added model config with defaults")
        elif "params" not in config["model"]:
            config["model"]["params"] = {}
            changes.append("Added model.params")
        
        return changes, warnings
    
    def _migrate_1_1_to_1_2(self, config: Dict) -> tuple:
        """Add training.epochs default."""
        changes = []
        warnings = []
        
        if "training" not in config:
            config["training"] = {
                "test_split": 0.2,
                "random_seed": 42,
                "shuffle": True,
                "epochs": 10
            }
            changes.append("Added training config with defaults")
        elif "epochs" not in config["training"]:
            config["training"]["epochs"] = 10
            changes.append("Added training.epochs=10 default")
        
        return changes, warnings
    
    def _migrate_1_2_to_1_3(self, config: Dict) -> tuple:
        """Add project_name and save_model."""
        changes = []
        warnings = []
        
        if "project_name" not in config:
            config["project_name"] = "experiment"
            changes.append("Added project_name default")
        
        if "save_model" not in config:
            config["save_model"] = False
            changes.append("Added save_model=False default")
        
        return changes, warnings
    
    def _migrate_1_3_to_2_0(self, config: Dict) -> tuple:
        """Add v2.0 timeline configuration."""
        changes = []
        warnings = []
        
        if "timeline" not in config:
            config["timeline"] = {
                "enabled": True,
                "max_samples": 100,
                "user_samples": None
            }
            changes.append("Added timeline config for Learning Timeline feature")
        
        # Ensure schema version is set
        config["schema_version"] = "2.0"
        changes.append("Set schema_version to 2.0")
        
        return changes, warnings


class RunMigrator:
    """
    Handles migration of existing .gradia_logs run directories.
    """
    
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.schema_migrator = SchemaMigrator()
    
    def needs_migration(self) -> bool:
        """Check if run directory needs migration."""
        config = self._load_config()
        if config is None:
            return False
        
        version = self.schema_migrator.detect_version(config)
        return version != SchemaVersion.current().value
    
    def migrate(self) -> MigrationResult:
        """Migrate the run directory to current schema."""
        config = self._load_config()
        
        if config is None:
            return MigrationResult(
                success=False,
                from_version="unknown",
                to_version=SchemaVersion.current().value,
                changes=[],
                warnings=["No config.yaml found in run directory"]
            )
        
        result = self.schema_migrator.migrate(config)
        
        if result.success:
            self._save_config(config)
            
            # Create migration marker file
            marker_path = self.run_dir / ".migrated"
            marker_path.write_text(f"Migrated from {result.from_version} to {result.to_version}")
        
        return result
    
    def _load_config(self) -> Optional[Dict]:
        """Load config.yaml from run directory."""
        config_path = self.run_dir / "config.yaml"
        
        if not config_path.exists():
            return None
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _save_config(self, config: Dict):
        """Save config.yaml to run directory."""
        config_path = self.run_dir / "config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)


def migrate_all_runs(base_dir: Path = None) -> List[MigrationResult]:
    """
    Migrate all run directories in .gradia_logs.
    
    Args:
        base_dir: Base directory containing .gradia_logs (default: cwd)
        
    Returns:
        List of migration results
    """
    if base_dir is None:
        base_dir = Path.cwd()
    
    logs_dir = base_dir / ".gradia_logs"
    results = []
    
    if not logs_dir.exists():
        return results
    
    for run_dir in logs_dir.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            migrator = RunMigrator(run_dir)
            if migrator.needs_migration():
                result = migrator.migrate()
                results.append(result)
                print(f"Migrated {run_dir.name}: {result.from_version} → {result.to_version}")
    
    return results


def ensure_v2_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure config has all v2.0 fields with sensible defaults.
    
    Use this when loading configs to guarantee v2 compatibility.
    """
    migrator = SchemaMigrator()
    migrator.migrate(config)
    return config
