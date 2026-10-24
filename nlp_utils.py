from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
import numpy as np

class SentenceTransformerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        # Load model only when needed to save resources if not fitting
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        
        # X is typically a pandas Series or numpy array of text
        if hasattr(X, 'iloc'):
            texts = X.tolist()
        else:
            texts = list(X)
            
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings
