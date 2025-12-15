from typing import Any, Dict, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .base import GradiaModel

class SklearnWrapper(GradiaModel):
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)
        if hasattr(X, "columns"):
            self.feature_names = list(X.columns)

    def partial_fit(self, X, y, **kwargs):
        # For SGD (true partial_fit)
        if hasattr(self.model, "partial_fit"):
            classes = kwargs.get('classes')
            if classes is not None:
                self.model.partial_fit(X, y, classes=classes)
            else:
                self.model.partial_fit(X, y)
                
        # For RandomForest (warm_start simulation)
        elif hasattr(self.model, "warm_start") and self.model.warm_start:
            # Increase estimators by 1 step
            self.model.n_estimators += 1
            self.model.fit(X, y)
            
        if hasattr(X, "columns"):
            self.feature_names = list(X.columns)

    @property
    def supports_iterative(self) -> bool:
        return hasattr(self.model, "partial_fit") or (hasattr(self.model, "warm_start") and self.model.warm_start)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> Optional[np.ndarray]:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.feature_names:
            return None
            
        importances = None
        if hasattr(self.model, "coef_"):
            # Linear models
            importances = np.abs(self.model.coef_)
            if importances.ndim > 1:
                importances = importances.mean(axis=0) # Multiclass avg
        elif hasattr(self.model, "feature_importances_"):
            # Utilities (Tree based)
            importances = self.model.feature_importances_
            
        if importances is not None:
            return dict(zip(self.feature_names, importances))
        return None

    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()

class ModelFactory:
    @staticmethod
    def create(model_type: str, task_type: str, params: Dict[str, Any] = {}) -> GradiaModel:
        # Standard Linear
        if model_type == 'linear':
            if task_type == 'classification':
                return SklearnWrapper(LogisticRegression(**params))
            else:
                return SklearnWrapper(LinearRegression(**params))
        
        # Random Forest
        elif model_type == 'random_forest':
            # Enable warm_start for iterative viz if not specified
            if 'warm_start' not in params:
                 params['warm_start'] = True
            if task_type == 'classification':
                return SklearnWrapper(RandomForestClassifier(**params))
            else:
                return SklearnWrapper(RandomForestRegressor(**params))

        # SGD (Iterative Linear)
        elif model_type == 'sgd':
            # Map optimizer/learning rate params from UI to sklearn args if needed
            # User might pass 'lr', sklearn uses 'eta0' + 'learning_rate'='constant'/'invscaling'
            # simplified normalization handled by CLI or here.
            # For MVP, assume params are already sklearn-compatible or clean them up.
            if task_type == 'classification':
                 return SklearnWrapper(SGDClassifier(**params))
            else:
                 return SklearnWrapper(SGDRegressor(**params))
        
        # MLP / CNN (Basic Neural Net)
        elif model_type in ['mlp', 'cnn']:
             from sklearn.neural_network import MLPClassifier, MLPRegressor
             if task_type == 'classification':
                 # hidden_layer_sizes default for simple MNIST-like
                 if 'hidden_layer_sizes' not in params:
                     params['hidden_layer_sizes'] = (100, 50)
                 return SklearnWrapper(MLPClassifier(warm_start=True, **params))
             else:
                 return SklearnWrapper(MLPRegressor(warm_start=True, **params))

        # Default fallback
        if task_type == 'classification':
             return SklearnWrapper(RandomForestClassifier(warm_start=True, **params))
        else:
             return SklearnWrapper(RandomForestRegressor(warm_start=True, **params))
