import numpy as np
import pandas as pd
from math import pi


def make_zone_columns(df, index=['date_cible', 'echeance'], columns=['zone']):
    new_df = df.pivot_table(index=index,columns=columns)
    new_df.columns = ["_".join([str(c) for c in col]) for col in new_df.columns.values]
    return new_df


def compute_valwind(u,v):
    return np.sqrt(np.square(u) + np.square(v))


def compute_dirwind(u,v):
    arct = np.arctan2(u,v)
    angle = (arct * 180 / pi + 180)
    return angle


def expand_calendarfeatures(df):
    df["cos_dayofyear"] =  np.cos(df.date_cible.dt.dayofyear.values / 365.25 * 2 * pi)
    df["sin_dayofyear"] =  np.sin(df.date_cible.dt.dayofyear.values / 365.25 * 2 * pi)
    df["cos_day"] = np.cos(df.hh_mm_cible.values / 24 * 2 * pi)
    df["sin_day"] = np.sin(df.hh_mm_cible.values / 24 * 2 * pi)
    df = df.join(pd.get_dummies(df.mois_cible, prefix="month")).drop(columns='mois_cible')
    return df


class QuantileScore():

    def __init__(self, q=0.5):
        self.q = q
        
    def __call__(self, y_true, y_pred, weight=1):
        return np.sum((y_pred - y_true) * (1 * (y_true < y_pred) - self.q)) * weight