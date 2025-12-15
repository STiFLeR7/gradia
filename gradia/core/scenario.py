import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Any

@dataclass
class Scenario:
    dataset_path: str
    target_column: str
    task_type: str  # 'classification' or 'regression'
    is_multiclass: bool = False
    class_count: int = 0
    features: List[str] = field(default_factory=list)
    recommended_model: str = "random_forest"

class ScenarioInferrer:
    """Infers the ML scenario (Task type, Target) from a dataset."""

    POSSIBLE_TARGET_NAMES = ['target', 'label', 'y', 'class', 'outcome', 'price', 'score']
    
    def infer(self, file_path: str, target_override: Optional[str] = None) -> Scenario:
        # Load a sample to infer types
        df = self._load_sample(file_path)
        
        target = target_override
        if not target:
            target = self._guess_target(df)
            
        if not target:
            raise ValueError(f"Could not infer target column for {file_path}. Please name one of {self.POSSIBLE_TARGET_NAMES} or provide config.")
            
        task_type, is_multiclass, count = self._infer_task_type(df[target])
        features = [c for c in df.columns if c != target]
        
        recommended_model = self._infer_model_recommendation(features)
        
        return Scenario(
            dataset_path=str(file_path),
            target_column=target,
            task_type=task_type,
            is_multiclass=is_multiclass,
            class_count=count,
            features=features,
            recommended_model=recommended_model
        )

    def _infer_model_recommendation(self, features: List[str]) -> str:
        # Heuristic 1: Check for pixel data (Fashion MNIST, MNIST, etc.)
        # If > 100 features and names contain 'pixel'
        if len(features) > 100:
            pixel_cols = [f for f in features if 'pixel' in f.lower()]
            if len(pixel_cols) > len(features) * 0.5:
                return "cnn"
                
        # Heuristic 2: Tabular default
        return "random_forest"

    def _load_sample(self, path: str, n_rows: int = 1000) -> pd.DataFrame:
        if path.endswith('.csv'):
            return pd.read_csv(path, nrows=n_rows)
        elif path.endswith('.parquet'):
            # Parquet doesn't support 'nrows' efficiently same as csv sometimes, 
            # but pandas read_parquet usually loads full. For large files we might need pyarrow.
            # For MVP assume fits in memory or use logic to limits.
            return pd.read_parquet(path).head(n_rows)
        else:
            raise ValueError("Unsupported format")

    def _guess_target(self, df: pd.DataFrame) -> Optional[str]:
        # 1. Exact Name match
        for name in self.POSSIBLE_TARGET_NAMES:
            if name in df.columns:
                return name
            if name.upper() in df.columns:
                return name.upper()
        
        # 2. Heuristic: Avoid ID/Date columns
        candidates = []
        for col in df.columns:
            lower = col.lower()
            if not any(x in lower for x in ['id', 'date', 'time', 'created_at', 'uuid', 'index']):
                candidates.append(col)
        
        if candidates:
            return candidates[-1]
            
        # 3. Last column fallback
        return df.columns[-1]

    def _infer_task_type(self, series: pd.Series):
        """
        Returns (task_type, is_multiclass, class_count)
        """
        # Heuristics:
        # If string/object -> Classification
        # If float -> Regression (unless low cardinality?)
        # If int -> Check cardinality. Low (<20) -> Classification. High -> Regression.
        
        unique_count = series.nunique()
        dtype = series.dtype
        
        if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            return 'classification', unique_count > 2, unique_count
            
        if pd.api.types.is_float_dtype(dtype):
            # If floats are actually integers (e.g. 1.0, 0.0), check that
            if series.apply(float.is_integer).all() and unique_count < 20:
                 return 'classification', unique_count > 2, unique_count
            return 'regression', False, 0
            
        if pd.api.types.is_integer_dtype(dtype):
            if unique_count < 20:  # Arbitrary threshold for MVP
                return 'classification', unique_count > 2, unique_count
            else:
                return 'regression', False, 0
                
        # Fallback
        return 'regression', False, 0
