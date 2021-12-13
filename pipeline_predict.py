from matplotlib.pyplot import thetagrids
import joblib
import pandas as pd
import numpy as np
import config as cfg

titanic_model = joblib.load('titanicPipeline.pkl')

def predict(X):
    X = X[cfg.FEATURES]
    predicts = titanic_model.predict(X.T)
    salida = np.exp(predicts)
    print(salida)
