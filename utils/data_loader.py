import pandas as pd

def load_data(file):
    """ Load CSV file into a Pandas DataFrame """
    return pd.read_csv(file)
