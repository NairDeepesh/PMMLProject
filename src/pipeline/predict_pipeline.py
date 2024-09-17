import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object
import sys

def inverse_log_transform(y):
    return np.expm1(y)

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_x_path = os.path.join('artifacts', 'preprocessor_x.pkl')
        self.preprocessor_y_path = os.path.join('artifacts', 'preprocessor_y.pkl')
    
    def predict(self, features):
        try:
            model = load_object(file_path=self.model_path)
            preprocessor_x = load_object(file_path=self.preprocessor_x_path)
            preprocessor_y = load_object(file_path=self.preprocessor_y_path)

            data_scaled = preprocessor_x.transform(features)
            preds = model.predict(data_scaled)
            preds_scaled = preprocessor_y.inverse_transform(preds.reshape(-1, 1)).flatten()
            preds = inverse_log_transform(preds_scaled)

            return preds
        
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self, time_in_cycles, T50, P30, Ps30, Nc, NRf, phi, BPR, W32, htBleed):
        self.time_in_cycles = time_in_cycles
        self.T50 = T50
        self.P30 = P30
        self.Ps30 = Ps30
        self.Nc = Nc
        self.NRf = NRf
        self.phi = phi
        self.BPR = BPR
        self.W32 = W32
        self.htBleed = htBleed

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "time_in_cycles": [self.time_in_cycles],
                "T50": [self.T50],
                "P30": [self.P30],
                "Ps30": [self.Ps30],
                "Nc": [self.Nc],
                "NRf": [self.NRf],
                "phi": [self.phi],
                "BPR": [self.BPR],
                "W32": [self.W32],
                "htBleed": [self.htBleed],
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
