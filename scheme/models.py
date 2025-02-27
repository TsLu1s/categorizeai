import numpy as np
from catboost import CatBoostClassifier

class CatBoost_NLPClassifier:
    """
    A wrapper class for CatBoost specifically designed for NLP classification tasks.
    
    This classifier automatically detects whether the task is binary or multiclass classification
    and configures CatBoost accordingly. It provides a scikit-learn-like interface for ease of use.
    
    Parameters:
        label_column (str): Name of the column containing the target labels
        eval_metric (str, optional): Metric to evaluate the model. Defaults to 'Precision'
        verbose (bool, optional): Whether to show training progress. Defaults to False
        iterations (int, optional): Number of trees to build. Defaults to 1000
        learning_rate (float, optional): Learning rate for gradient descent. Defaults to 0.03
        
    Attributes:
        model: The underlying CatBoost model instance
        problem_type (str): Type of classification problem ('binary' or 'multiclass')
        is_fitted_ (bool): Whether the model has been fitted
    """
    
    def __init__(self, 
                 label_column,
                 verbose=False, 
                 iterations=1000,
                 learning_rate=0.03):
        self.label_column = label_column
        self.verbose = verbose
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.model = None
        self.problem_type = None
        
    def _detect_problem_type(self, y):
        """
        Automatically detects whether the classification task is binary or multiclass.
        
        Parameters:
            y (array-like): Target labels
            
        Note:
            Sets the problem_type attribute to either 'binary' or 'multiclass'
        """
        unique_labels = np.unique(y)
        if len(unique_labels) == 2:
            self.problem_type = 'binary'
        else:
            self.problem_type = 'multiclass'
            
    def fit(self, train_data, eval_set=None):
        """
        Fits the CatBoost model on the training data.
        
        Parameters:
            train_data (pd.DataFrame): Training data including both features and label column
            eval_set (tuple, optional): (X_val, y_val) for validation during training
        
        Returns:
            self: The fitted classifier
            
        Note:
            The training data must include the label column specified during initialization
        """
        y = train_data[self.label_column]
        X = train_data.drop(columns=[self.label_column])
        
        self._detect_problem_type(y)
        
        # Configure CatBoost parameters based on problem type
        params = {
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'verbose': self.verbose,
            'loss_function': 'Logloss' if self.problem_type == 'binary' else 'MultiClass',        
            }
        
        self.model = CatBoostClassifier(**params)
        
        # Prepare validation set if provided
        if eval_set is not None:
            X_val, y_val = eval_set
            self.model.fit(X, y, eval_set=(X_val, y_val))
        else:
            self.model.fit(X, y)
        
        self.is_fitted_ = True
        return self
        
    def predict(self, test_data):
        """
        Generate predictions for test data.
        
        Parameters:
            test_data (pd.DataFrame): Test data with the same features as training data
            
        Returns:
            array-like: Predicted class labels
            
        Raises:
            AttributeError: If the model hasn't been fitted yet
        """
        self.check_is_fitted()
        if self.label_column in test_data.columns:
            test_data = test_data.drop(columns=[self.label_column])
        return self.model.predict(test_data)
    
    def predict_proba(self, test_data):
        """
        Generate probability predictions for test data.
        
        Parameters:
            test_data (pd.DataFrame): Test data with the same features as training data
            
        Returns:
            array-like: Predicted class probabilities. For binary classification,
                       returns probabilities for each class in a 2D array
            
        Raises:
            AttributeError: If the model hasn't been fitted yet
        """
        self.check_is_fitted()
        if self.label_column in test_data.columns:
            test_data = test_data.drop(columns=[self.label_column])
        return self.model.predict_proba(test_data)
    
    def check_is_fitted(self):
        """
        Checks if the model has been fitted.
        
        Raises:
            AttributeError: If the model hasn't been fitted yet
            
        Note:
            This method should be called at the beginning of predict and predict_proba
        """
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise AttributeError("Model not fitted yet. Call 'fit' first.")
    
    def save_model(self, path: str):
        """Save the model to disk."""
        if self.model is not None:
            self.model.save_model(path)
        else:
            raise ValueError("No model to save. Model must be fitted first.")
    
    def load_model(self, path: str):
        """Load a saved model from disk."""
        self.model = CatBoostClassifier()
        self.model.load_model(path)
        self.is_fitted_ = True
        return self
