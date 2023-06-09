import joblib
import pickle
import numpy as np
import pandas as pd
from scipy import stats

import plotly
from plotly import express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

### Converting to datetime
# df['DateTime'] = pd.to_datetime(df.DateTime, utc=True).dt.tz_convert('Etc/GMT-11')
def iterate_all(iterable, returned="key"):
    """Returns an iterator that returns all keys or values
       of a (nested) iterable.

       Arguments:
           - iterable: <list> or <dictionary>
           - returned: <string> "key" or "value"

       Returns:
           - <iterator>
    """

    if isinstance(iterable, dict):
        for key, value in iterable.items():
            if returned == "key":
                yield key
            elif returned == "value":
                if not (isinstance(value, dict) or isinstance(value, list)):
                    yield value
            else:
                raise ValueError("'returned' keyword only accepts 'key' or 'value'.")
            for ret in iterate_all(value, returned=returned):
                yield ret
    elif isinstance(iterable, list):
        for el in iterable:
            for ret in iterate_all(el, returned=returned):
                yield ret


def iterate_partial(parent_iterable:dict, up_to_child_iterables:[dict], returned:str='key', roots_of_child_to_exclude: list=[]):
    removed_child_iterables = []
    for child_iterable in up_to_child_iterables:
        removed_child_iterables += list(iterate_all(child_iterable, returned=returned))


    child_root_included = [item for item in list(iterate_all(parent_iterable, returned=returned)) if item not in removed_child_iterables]

    if len(roots_of_child_to_exclude) !=0:
        return [k for k in child_root_included if  k not in roots_of_child_to_exclude]
    return child_root_included


def prediction_Interval(res, model, exog_test, CI=0.95, mode="unbiased"):
    """
    Generate prediction interval for sm.OLS
    exog_test: test data with first column beign 1s
    """

    degrees_of_freedom = model.exog.shape[0] - model.exog.shape[1]
    RSS = res.ssr

    if mode == "unbiased":
        sigma = np.sqrt(RSS / degrees_of_freedom)
    elif mode == "ml":
        sigma = np.sqrt(RSS / model.exog.shape[0].shape[0])

    point_pred = res.predict(exog_test)

    return (point_pred - stats.norm.ppf(CI) * sigma, point_pred + stats.norm.ppf(CI) * sigma)


def fitted_residual_plot(fitted, residual, file_path=None):
    f = go.Figure()
    f.add_trace(go.Scatter(x=fitted, y=residual, mode='markers'))
    f.add_trace(go.Scatter(x=[fitted.min(), fitted.max()], y=[0, 0], mode="lines"))
    if file_path:
        f.write_html(file_path)
    else:
        f.show()

def predictor_residual_plot(predictor, residual, file_path=None):
    f = go.Figure()
    f.add_trace(go.Scatter(x=predictor, y=residual, mode='markers'))
    f.add_trace(go.Scatter(x=[predictor.min(), predictor.max()], y=[0, 0], mode="lines"))
    if file_path:
        f.write_html(file_path)
    else:
        f.show()


def generate_shift_day(df):
    df["hour"] = df.DateTime.dt.strftime("%Y-%m-%d ") + df.DateTime.dt.strftime("%H:00")

    out_series = pd.Series(index=df.index)
    out_series.loc[df.DateTime.dt.hour >= 7 ] = df.loc[df.DateTime.dt.hour >= 7].DateTime.dt.strftime("%Y-%m-%d 07:00:00")
    out_series.loc[df.DateTime.dt.hour < 7] = (df.loc[ df.DateTime.dt.hour < 7].DateTime - pd.to_timedelta(1, unit="day")).dt.strftime("%Y-%m-%d 07:00:00")

    df = df.drop('hour', axis=1)

    return out_series

def generate_shifts(df):
    df["hour"] = df.DateTime.dt.strftime("%Y-%m-%d ") + df.DateTime.dt.strftime("%H:00")

    out_series = pd.Series(index=df.index)

    out_series.loc[(df.DateTime.dt.hour >= 7) & (df.DateTime.dt.hour < 19)] = df.loc[(df.DateTime.dt.hour >= 7) & (df.DateTime.dt.hour < 19)].DateTime.dt.strftime("%Y-%m-%d 07:00:00")
    out_series.loc[(df.DateTime.dt.hour >= 19) & (df.DateTime.dt.hour < 24)] = df.loc[(df.DateTime.dt.hour >= 19) & (df.DateTime.dt.hour < 24)].DateTime.dt.strftime("%Y-%m-%d 19:00:00")
    out_series.loc[(df.DateTime.dt.hour >= 0) & (df.DateTime.dt.hour < 7)] = (df.loc[(df.DateTime.dt.hour >= 0) & (df.DateTime.dt.hour < 7)].DateTime - pd.to_timedelta(1, unit="day")).dt.strftime("%Y-%m-%d 19:00:00")

    df = df.drop('hour', axis=1)

    return out_series


def create_shifts(df):
    df["hour"] = df.DateTime.dt.strftime("%Y-%m-%d ") + df.DateTime.dt.strftime("%H:00")

    # generate shift_start_dates
    df["shift_start"] = np.NAN
    df.loc[(df.DateTime.dt.hour >= 7) & (df.DateTime.dt.hour < 19), "shift_start"] = df.loc[
        (df.DateTime.dt.hour >= 7) & (df.DateTime.dt.hour < 19)].DateTime.dt.strftime("%Y-%m-%d 07:00:00")
    df.loc[(df.DateTime.dt.hour >= 19) & (df.DateTime.dt.hour < 24), "shift_start"] = df.loc[
        (df.DateTime.dt.hour >= 19) & (df.DateTime.dt.hour < 24)].DateTime.dt.strftime("%Y-%m-%d 19:00:00")
    df.loc[(df.DateTime.dt.hour >= 0) & (df.DateTime.dt.hour < 7), "shift_start"] = (
            df.loc[(df.DateTime.dt.hour >= 0) & (df.DateTime.dt.hour < 7)].DateTime - pd.to_timedelta(1, unit="day")).dt.strftime("%Y-%m-%d 19:00:00")

    df = df.drop('hour', axis=1)

    return df

def cross_correlation_search(df,lag_col_name,col_name2, lag_periods=np.arange(-10,10), plot=False):
     """
     - method options:  pearson : Standard correlation coefficient -- Linear Relationships
                        kendall : Kendall Tau correlation coefficient -- Measure of rank correlation - Good for ordinal var.
                        spearman : Spearman rank correlation --
     """
     corr_coefs= []
     for lag in lag_periods:
          corr_coefs.append((df[lag_col_name].shift(lag)).corr(df[col_name2]))


     if plot:
          f=px.scatter(x=lag_periods, y = corr_coefs)
          f.show()

     return dict(zip(lag_periods,corr_coefs))

# Save the object
def save_joblib(obj, file_name):
    with open(file_name, 'wb') as file:
       joblib.dump(obj, file)
       file.close()
       print(f"Saved the object to {file_name} successfully!")

def load_joblib(file_name):
    with open(file_name, 'rb') as file:
       obj = joblib.load(file)
       file.close()
       return obj


def save_pickle(obj, file_name):
    """
    Save a python object as a .pkl file
    """
    with open(file_name, mode="wb") as file:
        pickle.dump(obj, file)
        file.close()
        print(f"Saved the object to {file_name} successfully!")


# Load the object
def load_pickle(file_name):
    """
    Load a .pkl file
    """
    with open(file_name, mode="rb") as file:
        obj = pickle.load(file)
        file.close()
        return obj