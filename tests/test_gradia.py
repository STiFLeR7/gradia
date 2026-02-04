"""
Gradia v2.0.0 Test Suite

Tests for core functionality, events module, timeline tracking, and API endpoints.
"""

import os
import shutil
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient
import time

# Import Application Code
from gradia.core.inspector import Inspector
from gradia.core.scenario import ScenarioInferrer, Scenario
from gradia.core.config import ConfigManager
from gradia.core.migration import SchemaMigrator, SchemaVersion, ensure_v2_config
from gradia.viz.server import app
from gradia.models.sklearn_wrappers import ModelFactory
from gradia.events import LearningEvent, SampleTracker, TimelineLogger
from gradia.events.models import SampleState, EpochSummary

# Setup
TEST_CSV = "test_data.csv"
TEST_RUN_DIR = Path("test_runs")


@pytest.fixture(scope="module")
def setup_environment():
    """Create test environment with dummy CSV data."""
    # 1. Create Dummy CSV with clear classification target
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100),
        'label': np.random.randint(0, 3, 100)  # Multiclass classification
    })
    df.to_csv(TEST_CSV, index=False)
    
    # Clean previous test runs
    if TEST_RUN_DIR.exists():
        shutil.rmtree(TEST_RUN_DIR)
    TEST_RUN_DIR.mkdir()
    
    yield
    
    # Cleanup
    if os.path.exists(TEST_CSV):
        try:
            os.remove(TEST_CSV)
        except Exception:
            pass
    if TEST_RUN_DIR.exists():
        try:
            shutil.rmtree(TEST_RUN_DIR)
        except Exception:
            pass


# =============================================================================
# Core Module Tests
# =============================================================================

class TestInspector:
    """Tests for dataset discovery."""
    
    def test_find_datasets(self, setup_environment):
        """Verify Inspector finds CSV files."""
        inspector = Inspector(Path("."))
        datasets = inspector.find_datasets()
        
        test_ds = next((d for d in datasets if d.name == TEST_CSV), None)
        assert test_ds is not None, f"{TEST_CSV} not found. Found: {[d.name for d in datasets]}"
    
    def test_supported_extensions(self):
        """Verify supported file types."""
        assert '.csv' in Inspector.SUPPORTED_EXTENSIONS
        assert '.parquet' in Inspector.SUPPORTED_EXTENSIONS


class TestScenarioInference:
    """Tests for scenario detection."""
    
    def test_basic_inference(self, setup_environment):
        """Verify task type detection from CSV."""
        inferrer = ScenarioInferrer()
        scenario = inferrer.infer(TEST_CSV, target_override='label')
        
        assert scenario.task_type == 'classification'
        assert scenario.target_column == 'label'
        assert 'feature1' in scenario.features
        assert 'feature2' in scenario.features
        assert 'label' not in scenario.features
    
    def test_multiclass_detection(self, setup_environment):
        """Verify multiclass classification detection."""
        inferrer = ScenarioInferrer()
        scenario = inferrer.infer(TEST_CSV, target_override='label')
        
        assert scenario.is_multiclass == True
        assert scenario.class_count == 3
    
    def test_model_recommendation(self, setup_environment):
        """Verify model suggestions."""
        inferrer = ScenarioInferrer()
        scenario = inferrer.infer(TEST_CSV, target_override='label')
        
        # Small tabular dataset should suggest random_forest
        assert scenario.recommended_model in ['random_forest', 'svm', 'mlp', 'cnn', 'logistic']


class TestConfigManager:
    """Tests for configuration management."""
    
    def test_default_config_v2(self, setup_environment):
        """Verify v2 default config structure."""
        config_mgr = ConfigManager(TEST_RUN_DIR)
        config = config_mgr.load_or_create()
        
        # v2 required fields
        assert 'schema_version' in config
        assert config['schema_version'] == '2.0'
        assert 'timeline' in config
        assert config['timeline']['enabled'] == True
        assert config['timeline']['max_samples'] == 100
    
    def test_config_save_load(self, setup_environment):
        """Verify config persistence."""
        config_mgr = ConfigManager(TEST_RUN_DIR)
        config = config_mgr.load_or_create()
        config['project_name'] = 'test_project'
        config_mgr.save(config)
        
        # Load again
        config_mgr2 = ConfigManager(TEST_RUN_DIR)
        loaded = config_mgr2.load_or_create()
        
        assert loaded['project_name'] == 'test_project'
        assert loaded['schema_version'] == '2.0'


# =============================================================================
# Migration Tests
# =============================================================================

class TestMigration:
    """Tests for schema migration."""
    
    def test_version_detection_v1(self):
        """Detect v1.0 config."""
        migrator = SchemaMigrator()
        old_config = {'model': {'type': 'rf'}}
        
        version = migrator.detect_version(old_config)
        assert version == '1.0'
    
    def test_version_detection_v2(self):
        """Detect v2.0 config."""
        migrator = SchemaMigrator()
        new_config = {'schema_version': '2.0', 'timeline': {'enabled': True}}
        
        version = migrator.detect_version(new_config)
        assert version == '2.0'
    
    def test_migration_v1_to_v2(self):
        """Migrate v1.0 config to v2.0."""
        migrator = SchemaMigrator()
        old_config = {
            'model': {'type': 'random_forest'},
            'training': {'test_split': 0.2}
        }
        
        result = migrator.migrate(old_config)
        
        assert result.success == True
        assert result.from_version == '1.0'
        assert result.to_version == '2.0'
        assert 'timeline' in old_config
        assert old_config['schema_version'] == '2.0'
    
    def test_no_migration_needed(self):
        """Skip migration for current version."""
        migrator = SchemaMigrator()
        config = {'schema_version': '2.0', 'timeline': {'enabled': True}}
        
        result = migrator.migrate(config)
        
        assert result.success == True
        assert 'No migration needed' in result.changes


# =============================================================================
# Events Module Tests (v2.0)
# =============================================================================

class TestLearningEvent:
    """Tests for LearningEvent model."""
    
    def test_event_creation(self):
        """Create and serialize LearningEvent."""
        event = LearningEvent(
            run_id="test_run",
            epoch=1,
            sample_id=42,
            true_label=1,
            predicted_label=1,
            confidence=0.95,
            correct=True
        )
        
        assert event.run_id == "test_run"
        assert event.epoch == 1
        assert event.correct == True
        
        # Serialization
        d = event.to_dict()
        assert d['sample_id'] == 42
        
        # Deserialization
        event2 = LearningEvent.from_dict(d)
        assert event2.confidence == 0.95


class TestSampleState:
    """Tests for SampleState tracking."""
    
    def test_flip_count(self):
        """Count prediction flips."""
        state = SampleState(sample_id=0, true_label=1)
        
        # Add events with changing predictions
        for epoch, pred in enumerate([0, 1, 1, 0, 1], start=1):
            event = LearningEvent(
                run_id="test", epoch=epoch, sample_id=0,
                true_label=1, predicted_label=pred,
                confidence=0.8, correct=(pred == 1)
            )
            state.add_event(event)
        
        assert state.flip_count == 3  # 0→1, 1→0, 0→1
    
    def test_stability_classification(self):
        """Test stability class detection."""
        # Stable correct
        state = SampleState(sample_id=0, true_label=1)
        for epoch in range(1, 6):
            state.add_event(LearningEvent(
                run_id="test", epoch=epoch, sample_id=0,
                true_label=1, predicted_label=1,
                confidence=0.9, correct=True
            ))
        
        assert state.stability_class == "stable_correct"
        assert state.is_stable == True


class TestSampleTracker:
    """Tests for SampleTracker."""
    
    def test_initialization(self, setup_environment):
        """Initialize tracker with data."""
        X = np.random.rand(50, 3)
        y = np.array([0, 1] * 25)
        
        tracker = SampleTracker(max_samples=20, seed=42)
        tracker.initialize(X, y, run_id="test_run")
        
        assert len(tracker.tracked_indices) <= 20
        assert tracker._initialized == True
    
    def test_record_predictions(self, setup_environment):
        """Record and retrieve predictions."""
        X = np.random.rand(50, 3)
        y = np.array([0, 1] * 25)
        preds = np.array([0, 1] * 25)
        
        tracker = SampleTracker(max_samples=10, seed=42)
        tracker.initialize(X, y, run_id="test_run")
        
        events = tracker.record_predictions(
            epoch=1, X=X, y=y,
            predictions=preds, probabilities=None
        )
        
        assert len(events) == 10
        assert all(isinstance(e, LearningEvent) for e in events)
    
    def test_epoch_summary(self, setup_environment):
        """Generate epoch summary."""
        X = np.random.rand(50, 3)
        y = np.array([0, 1] * 25)
        preds = y.copy()  # Perfect predictions
        
        tracker = SampleTracker(max_samples=10, seed=42)
        tracker.initialize(X, y, run_id="test_run")
        tracker.record_predictions(epoch=1, X=X, y=y, predictions=preds)
        
        summary = tracker.get_epoch_summary(epoch=1)
        
        assert isinstance(summary, EpochSummary)
        assert summary.epoch == 1
        assert summary.total_tracked == 10


class TestTimelineLogger:
    """Tests for TimelineLogger."""
    
    def test_log_and_read_events(self, setup_environment):
        """Log and retrieve events."""
        logger = TimelineLogger(str(TEST_RUN_DIR))
        logger.clear()
        
        event = LearningEvent(
            run_id="test", epoch=1, sample_id=0,
            true_label=1, predicted_label=1,
            confidence=0.9, correct=True
        )
        
        logger.log_event(event)
        logger.finalize()
        
        events = logger.get_events()
        assert len(events) == 1
        assert events[0].sample_id == 0
    
    def test_filter_by_epoch(self, setup_environment):
        """Filter events by epoch."""
        logger = TimelineLogger(str(TEST_RUN_DIR))
        logger.clear()
        
        for epoch in [1, 2, 3]:
            event = LearningEvent(
                run_id="test", epoch=epoch, sample_id=0,
                true_label=1, predicted_label=1,
                confidence=0.9, correct=True
            )
            logger.log_event(event)
        
        logger.finalize()
        
        epoch_2_events = logger.get_events(epoch=2)
        assert len(epoch_2_events) == 1
        assert epoch_2_events[0].epoch == 2


# =============================================================================
# Model Tests
# =============================================================================

class TestModelFactory:
    """Tests for model creation."""
    
    def test_create_random_forest(self):
        """Create RandomForest model."""
        model = ModelFactory.create('random_forest', 'classification', {'n_estimators': 10})
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_create_linear(self):
        """Create linear model."""
        model = ModelFactory.create('linear', 'regression', {})
        assert model is not None
    
    def test_iterative_support(self):
        """Check iterative training support."""
        rf_model = ModelFactory.create('random_forest', 'classification', {'warm_start': True})
        assert rf_model.supports_iterative == True
        
        sgd_model = ModelFactory.create('sgd', 'classification', {})
        assert sgd_model.supports_iterative == True


# =============================================================================
# API Tests
# =============================================================================

class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""
    
    def test_root_redirect(self, setup_environment):
        """Root redirects to configure when no trainer."""
        client = TestClient(app)
        response = client.get("/", follow_redirects=False)
        
        assert response.status_code == 307
        assert response.headers["location"] == "/configure"
    
    def test_configure_page(self, setup_environment):
        """Configure page loads with scenario."""
        client = TestClient(app)
        
        from gradia.viz import server
        
        # Inject state
        inferrer = ScenarioInferrer()
        scenario = inferrer.infer(TEST_CSV, target_override='label')
        
        server.SCENARIO = scenario
        server.CONFIG_MGR = ConfigManager(TEST_RUN_DIR)
        server.DEFAULT_CONFIG = server.CONFIG_MGR.load_or_create()
        server.RUN_DIR = TEST_RUN_DIR
        
        response = client.get("/configure")
        
        assert response.status_code == 200
        assert "Configure Experiment" in response.text
    
    def test_timeline_page_redirect(self, setup_environment):
        """Timeline page redirects without trainer."""
        client = TestClient(app)
        
        from gradia.viz import server
        
        # Setup scenario
        inferrer = ScenarioInferrer()
        server.SCENARIO = inferrer.infer(TEST_CSV, target_override='label')
        server.CONFIG_MGR = ConfigManager(TEST_RUN_DIR)
        server.DEFAULT_CONFIG = server.CONFIG_MGR.load_or_create()
        server.RUN_DIR = TEST_RUN_DIR
        server.TRAINER = None  # No trainer yet
        
        # Without trainer, should redirect
        response = client.get("/timeline", follow_redirects=False)
        assert response.status_code == 307
    
    def test_api_report_json(self, setup_environment):
        """JSON report endpoint."""
        client = TestClient(app)
        
        response = client.get("/api/report/json")
        # May be 404 if no logs, but shouldn't crash
        assert response.status_code in [200, 404]
    
    def test_api_start_training(self, setup_environment):
        """Start training via API."""
        client = TestClient(app)
        
        from gradia.viz import server
        
        # Setup
        inferrer = ScenarioInferrer()
        server.SCENARIO = inferrer.infer(TEST_CSV, target_override='label')
        server.CONFIG_MGR = ConfigManager(TEST_RUN_DIR)
        server.DEFAULT_CONFIG = server.CONFIG_MGR.load_or_create()
        server.RUN_DIR = TEST_RUN_DIR
        
        payload = {
            "model": {"type": "random_forest", "params": {"n_estimators": 5}},
            "training": {"epochs": 2},
            "project_name": "test_v2",
            "save_model": False
        }
        
        response = client.post("/api/start", json=payload)
        
        assert response.status_code == 200
        assert response.json() == {"status": "started"}
        
        # Wait for training to initialize
        time.sleep(1.5)
        
        assert server.TRAINER is not None
        assert server.TRAINING_THREAD is not None


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_training_flow(self, setup_environment):
        """Complete training flow with timeline."""
        from gradia.trainer.engine import Trainer
        
        # Setup scenario
        inferrer = ScenarioInferrer()
        scenario = inferrer.infer(TEST_CSV, target_override='label')
        
        # Create config
        run_path = TEST_RUN_DIR / "integration_test"
        config_mgr = ConfigManager(run_path)
        config = config_mgr.load_or_create()
        config['model']['type'] = 'random_forest'
        config['model']['params'] = {'n_estimators': 5}
        config['training']['epochs'] = 3
        config['timeline']['enabled'] = True
        config['timeline']['max_samples'] = 20
        
        # Create and run trainer
        trainer = Trainer(scenario, config, str(run_path))
        
        # Run should complete without error
        trainer.run()
        
        # Verify timeline was created
        assert trainer.enable_timeline == True
        assert trainer.sample_tracker is not None
        assert len(trainer.sample_tracker.tracked_indices) <= 20
    
    def test_evaluate_full(self, setup_environment):
        """Test full evaluation with timeline insights."""
        from gradia.trainer.engine import Trainer
        
        # Setup
        inferrer = ScenarioInferrer()
        scenario = inferrer.infer(TEST_CSV, target_override='label')
        
        run_path = TEST_RUN_DIR / "eval_test"
        config_mgr = ConfigManager(run_path)
        config = config_mgr.load_or_create()
        config['model']['type'] = 'random_forest'
        config['model']['params'] = {'n_estimators': 5}
        config['training']['epochs'] = 2
        
        trainer = Trainer(scenario, config, str(run_path))
        trainer.run()
        
        # Evaluate
        results = trainer.evaluate_full()
        
        assert 'accuracy' in results or 'mse' in results
        if scenario.task_type == 'classification':
            assert 'confusion_matrix' in results
            assert 'timeline_insights' in results
