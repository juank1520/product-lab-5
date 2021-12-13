import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

# Clase para el manejo de variables temporales en el modelo de House Price
class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables, reference_variables):
        if not isinstance(variables, list):
            raise ValueError('La variable debe de ser incluida en una lista')
        
        self.variables = variables
        self.reference_variables = reference_variables

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        x = x.copy()
        for feature in self.variables:
            x[feature] = x[self.reference_variables] - x[feature]
        return x

class Mapper(BaseEstimator, TransformerMixin):
    def __init__(self, variables, mappings):
        if not isinstance(variables, list):
            raise ValueError('La variable debe de ser incluida en una lista')
        
        self.variables = variables
        self.mappings = mappings
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        x = x.copy()
        for variable in self.variables:
            x[variable] = x[variable].map(self.mappings)
        return x

