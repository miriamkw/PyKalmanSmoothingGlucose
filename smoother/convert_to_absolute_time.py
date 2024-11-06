import pandas as pd

def convert_to_absolute_time(reltimes, startdatetime):
    datetimes = [startdatetime + pd.Timedelta(minutes=rel) for rel in reltimes]

    return pd.DatetimeIndex(datetimes)


