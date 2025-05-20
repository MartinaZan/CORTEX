from src.core.oracle_base import Oracle 

import numpy as np

class PerfectOracle(Oracle):

    def init(self):
        super().init()
        self.model = ""

    def real_fit(self):
        pass

    def _real_predict(self, data_instance):
        return data_instance.label
        
    def _real_predict_proba(self, data_instance):
        # softmax-style probability predictions
        if data_instance.label == 0:
            return np.array([1,0])
        else:
            return np.array([0,1])