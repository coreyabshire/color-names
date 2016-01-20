import pandas as pd

def read_old_data(filename):
    data = pd.read_csv(filename)
    data = data.ix[0:len(data)-5,:]
    all_match = data.ix[:, 5:8].apply(lambda x: min(x) == max(x), 1)
    data = data[all_match]
    coords = data.iloc[:, 2:5]
    names = data.iloc[:, 5]
    assert isinstance(coords, pd.DataFrame)
    return coords, names
