# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 17:50:10 2022

@author: heckenna
"""

import numpy as np
import pandas as pd
import data_opener


data = data_opener.get_data()
X_train, X_test, X_val, y_train, y_test, y_val = data_opener.train_test_val_split(data)



