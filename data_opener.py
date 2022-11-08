# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 09:43:18 2022

@author: heckenna
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# This may not quite be the function we want, be we will roll with it for now
def train_test_val_split(data, target = "cardio"):
    # Target is cardio
    target = "cardio"
    
    targ = data[target]
    features = data.drop(columns = target)
    
    # For 60-20-20 split... may not want to do here for CV, but we can fix later if needed
    X_train, X_test, y_train, y_test = train_test_split(features, targ, test_size=0.4, random_state=9208)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=9873)

    return X_train, X_test, X_val, y_train, y_test, y_val

def get_data(filename = "cardio_train.csv"):
    df = pd.read_csv(filename, sep = ";")
    
    df.dropna(inplace = True) # As it turns out, we have no NaNs. Superb!!!
    
    # Get rid of id column. Dont want to train on that!
    df.drop(columns = "id", inplace = True)
    
    # We have 2 ordinal variables: Cholesterol and Glucose.
    # Going to use one hot encoding on them
    df_dummied = pd.get_dummies(df, columns = ["cholesterol", "gluc", "gender"], drop_first = True)
    
    # Renames at least the gender column to have meaning. 
    # Could do same to chol and gluc if desired
    df_dummied.rename(columns = {"gender_2":"is_man"}, inplace = True)
    
    return df_dummied


#X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(df_dummied)











