# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users/Dell/Downloads/sales forecasting 2/deploy/trained_model.sav', 'rb'))

input_data = (141.6180,9.0,1.0,1.0,24)
numpy_array = np.asarray(input_data)
reshaped = numpy_array.reshape(1,-1)
pred = loaded_model.predict(reshaped)[0]
print(pred)
print(f"Sales Value is between {pred-714.42} and {pred+714.42}")