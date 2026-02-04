from typing import Any, Dict, List
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, mean_absolute_error, confusion_matrix
)
from ..models.base import GradiaModel
from ..models.sklearn_wrappers import ModelFactory
from ..core.scenario import Scenario
from .callbacks import Callback, EventLogger
from ..events import SampleTracker, TimelineLogger

class Trainer:
    def __init__(self, scenario: Scenario, config: Dict[str, Any], run_dir: str):
        self.scenario = scenario
        self.config = config
        self.run_dir = run_dir
        print(f"DEBUG: Trainer initialized with RUN_DIR: {self.run_dir}")
        self.model: GradiaModel = ModelFactory.create(
            config['model']['type'], 
            scenario.task_type,
            config['model'].get('params', {})
        )
        self.callbacks: List[Callback] = [EventLogger(run_dir)]
        
        # v2.0: Learning Timeline support
        self.enable_timeline = config.get('timeline', {}).get('enabled', True)
        self.sample_tracker: SampleTracker = None
        self.timeline_logger: TimelineLogger = None
        
        if self.enable_timeline:
            timeline_config = config.get('timeline', {})
            self.sample_tracker = SampleTracker(
                max_samples=timeline_config.get('max_samples', 100),
                seed=config['training'].get('random_seed', 42)
            )
            self.timeline_logger = TimelineLogger(run_dir)
        
        # Store data references for evaluation
        self._X_test = None
        self._y_test = None
        self._X_train = None
        self._y_train = None

    def run(self):
        print("DEBUG: Trainer.run() started.")
        try:
            # 1. Load Data
            df = self._load_full_data()
            
            # 2. Preprocess
            df = df.dropna()
            
            # Separate Target and Features
            y = df[self.scenario.target_column]
            X = df[self.scenario.features]
            
            # --- Robust Preprocessing ---
            # 1. Identify non-numeric columns
            non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            cols_to_drop = []
            cols_to_encode = []
            
            for col in non_numeric_cols:
                # high cardinality heuristic (>50 unique and >5% of data) => drop (likely ID or Name)
                unique_count = X[col].nunique()
                if unique_count > 50 and unique_count > len(X) * 0.05:
                    cols_to_drop.append(col)
                else:
                    cols_to_encode.append(col)
            
            if cols_to_drop:
                X = X.drop(columns=cols_to_drop)
                print(f"Dropped high-cardinality/ID columns: {cols_to_drop}")
                
            # 2. One-Hot Encode
            if cols_to_encode:
                X = pd.get_dummies(X, columns=cols_to_encode, drop_first=True)
                print(f"Encoded columns: {cols_to_encode}")
                
            # Update features list for the UI
            self.scenario.features = X.columns.tolist()
            
            # --- End Preprocessing ---
            
            # Simple encoding for classification target if string
            if self.scenario.task_type == 'classification' and y.dtype == 'object':
                y = y.astype('category').cat.codes
                
            # 3. Split
            test_size = self.config['training'].get('test_split', 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, 
                random_state=self.config['training'].get('random_seed', 42)
            )
            
            # Store for evaluation
            self._X_train = X_train
            self._y_train = y_train
            self._X_test = X_test
            self._y_test = y_test
            
            # v2.0: Initialize sample tracker for Learning Timeline
            run_id = f"run_{int(time.time())}"
            if self.enable_timeline and self.sample_tracker:
                user_samples = self.config.get('timeline', {}).get('user_samples', None)
                self.sample_tracker.initialize(
                    X=X_test.values if hasattr(X_test, 'values') else X_test,
                    y=y_test.values if hasattr(y_test, 'values') else y_test,
                    run_id=run_id,
                    user_indices=user_samples
                )
                # Clear previous timeline logs for fresh run
                self.timeline_logger.clear()
            
            # Notify Start
            epochs = self.config['training'].get('epochs', 10)
            self._dispatch('on_train_begin', {
                "scenario": str(self.scenario),
                "samples": len(df),
                "features": self.scenario.features,
                "epochs": epochs
            })
            
            # 4. Fit Loop
            epochs = self.config['training'].get('epochs', 10)
            
            if self.model.supports_iterative:
                # For Random Forest in warm_start, we need to set initial estimators
                if hasattr(self.model.model, "n_estimators"):
                    # We will grow it from 0 to 'epochs' (which acts as total estimators here)
                    self.model.model.n_estimators = 0
                
                classes = np.unique(y) if self.scenario.task_type == 'classification' else None
                
                # TQDM Output to Console
                with tqdm(range(1, epochs + 1), desc="Training", unit="epoch", colour="green") as pbar:
                    for epoch in pbar:
                        # Small delay to visualize speed if too fast
                        time.sleep(0.1) 
                        
                        self.model.partial_fit(X_train, y_train, classes=classes)
                        
                        # Evaluate
                        metrics = self._evaluate(X_train, y_train, X_test, y_test)
                        
                        # v2.0: Record timeline events
                        self._record_timeline_epoch(epoch, X_test, y_test)
                        
                        self._dispatch('on_epoch_end', epoch, metrics)
    
            else:
                # Non-Iterative Models (SVM, KNN, DecisionTree, etc.)
                # We fit once, then simulate "epochs" for user visual satisfaction
                print("Training standard model (single batch fits)...")
                self.model.fit(X_train, y_train)
                
                # Compute final metrics
                metrics = self._evaluate(X_train, y_train, X_test, y_test)
                
                # Simulate progress bar so UI doesn't look broken
                with tqdm(range(1, epochs + 1), desc="Training", unit="epoch", colour="blue") as pbar:
                    for epoch in pbar:
                        time.sleep(0.1) # Simulate work
                        # We broadcast the SAME metrics for every "epoch" since the model doesn't change
                        # But it keeps the UI happy and consistent
                        
                        # v2.0: Record timeline events (same predictions each epoch for non-iterative)
                        self._record_timeline_epoch(epoch, X_test, y_test)
                        
                        self._dispatch('on_epoch_end', epoch, metrics)
                        
                        # Update Progress Bar
                        pf = {}
                        if 'train_acc' in metrics:
                            pf['acc'] = f"{metrics['train_acc']:.3f}"
                        if 'train_mse' in metrics:
                            pf['mse'] = f"{metrics['train_mse']:.3f}"
                        pbar.set_postfix(pf)
                    
    
            
            # 5. Finalize
            fi = self.model.get_feature_importance()
            
            # v2.0: Finalize timeline logging
            if self.enable_timeline and self.timeline_logger:
                self.timeline_logger.save_tracker_state(self.sample_tracker.to_dict())
                self.timeline_logger.finalize()
        
            # 6. Training Complete
            timeline_summary = None
            if self.enable_timeline and self.sample_tracker:
                timeline_summary = {
                    "tracked_samples": len(self.sample_tracker.tracked_indices),
                    "top_flipping": [s.sample_id for s in self.sample_tracker.get_top_flipping_samples(5)],
                    "never_correct": len(self.sample_tracker.get_never_correct())
                }
            
            self._dispatch("on_train_end", {
                "epoch": epochs, 
                "feature_importance": self.model.get_feature_importance(),
                "timeline_summary": timeline_summary
            })
            
            # 7. Save Model
            if self.config.get('save_model'):
                ckpt_dir = Path(self.run_dir) / "models" / "best-ckpt"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / "model.pkl"
                with open(ckpt_path, "wb") as f:
                    pickle.dump(self.model, f)
                print(f"Model saved to {ckpt_path}")

            print(f"DEBUG: Trainer.run() finished successfully.")
            
        except Exception as e:
            print(f"CRITICAL ERROR IN TRAINER: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def _evaluate(self, X_train, y_train, X_test, y_test):
        preds_train = self.model.predict(X_train)
        preds_test = self.model.predict(X_test)
        
        metrics = {}
        if self.scenario.task_type == 'classification':
            metrics['train_acc'] = accuracy_score(y_train, preds_train)
            metrics['test_acc'] = accuracy_score(y_test, preds_test)
            
            # Weighted average for multiclass support
            metrics['precision'] = precision_score(y_test, preds_test, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, preds_test, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_test, preds_test, average='weighted', zero_division=0)
            
        else:
            metrics['train_mse'] = mean_squared_error(y_train, preds_train)
            metrics['test_mse'] = mean_squared_error(y_test, preds_test)
            metrics['mae'] = mean_absolute_error(y_test, preds_test)
            metrics['r2'] = r2_score(y_test, preds_test) 
            
        return metrics

    def _load_full_data(self):
        # MVP: Load everything into memory
        path = self.scenario.dataset_path
        if path.endswith('.csv'):
            return pd.read_csv(path)
        return pd.read_parquet(path)

    def _dispatch(self, method_name, *args, **kwargs):
        for cb in self.callbacks:
            getattr(cb, method_name)(*args, **kwargs)

    def _record_timeline_epoch(self, epoch: int, X_test, y_test):
        """
        Record sample-level predictions for Learning Timeline (v2.0).
        
        Captures predictions for all tracked samples and logs events.
        """
        if not self.enable_timeline or not self.sample_tracker:
            return
        
        try:
            # Get predictions for all test samples
            X_arr = X_test.values if hasattr(X_test, 'values') else X_test
            y_arr = y_test.values if hasattr(y_test, 'values') else y_test
            
            predictions = self.model.predict(X_arr)
            
            # Get probabilities if available
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                try:
                    probabilities = self.model.predict_proba(X_arr)
                except Exception:
                    pass
            
            # Record predictions for tracked samples
            events = self.sample_tracker.record_predictions(
                epoch=epoch,
                X=X_arr,
                y=y_arr,
                predictions=predictions,
                probabilities=probabilities
            )
            
            # Log events
            self.timeline_logger.log_events(events, flush=(epoch % 5 == 0))
            
            # Log epoch summary
            summary = self.sample_tracker.get_epoch_summary(epoch)
            self.timeline_logger.log_summary(summary)
            
        except Exception as e:
            print(f"Warning: Timeline recording failed for epoch {epoch}: {e}")

    def evaluate_full(self) -> Dict[str, Any]:
        """
        Run full evaluation on test set with confusion matrix.
        
        Returns:
            Dictionary with evaluation results including confusion matrix.
        """
        if self._X_test is None or self._y_test is None:
            raise ValueError("No test data available. Run training first.")
        
        results = {}
        
        preds = self.model.predict(self._X_test)
        
        if self.scenario.task_type == 'classification':
            results['accuracy'] = accuracy_score(self._y_test, preds)
            results['precision'] = precision_score(self._y_test, preds, average='weighted', zero_division=0)
            results['recall'] = recall_score(self._y_test, preds, average='weighted', zero_division=0)
            results['f1'] = f1_score(self._y_test, preds, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(self._y_test, preds)
            results['confusion_matrix'] = cm.tolist()
            
            # Class labels
            classes = sorted(list(set(self._y_test.tolist() if hasattr(self._y_test, 'tolist') else self._y_test)))
            results['classes'] = [str(c) for c in classes]
        else:
            results['mse'] = mean_squared_error(self._y_test, preds)
            results['mae'] = mean_absolute_error(self._y_test, preds)
            results['r2'] = r2_score(self._y_test, preds)
        
        # v2.0: Include timeline insights if available
        if self.enable_timeline and self.sample_tracker:
            results['timeline_insights'] = {
                'total_tracked': len(self.sample_tracker.tracked_indices),
                'top_flipping': [
                    {
                        'sample_id': s.sample_id,
                        'flip_count': s.flip_count,
                        'true_label': str(s.true_label),
                        'stability': s.stability_class
                    }
                    for s in self.sample_tracker.get_top_flipping_samples(10)
                ],
                'never_correct_count': len(self.sample_tracker.get_never_correct()),
                'late_learners_count': len(self.sample_tracker.get_late_learners())
            }
        
        return results

    def get_sample_timeline(self, sample_id: int) -> List[Dict[str, Any]]:
        """
        Get the learning timeline for a specific sample.
        
        Args:
            sample_id: Index of the sample to retrieve
            
        Returns:
            List of event dictionaries showing prediction evolution
        """
        if not self.enable_timeline or not self.sample_tracker:
            return []
        
        state = self.sample_tracker.get_sample_state(sample_id)
        if not state:
            return []
        
        return [event.to_dict() for event in state.history]
