import pandas as pd
import os

def process():
    _dir = os.path.json(os.path.dirname(__file__), "dataset/gauravdhamane/gwa-bitbrains/versions/1/fastStorage/2013-8")

    files = [f for f in os.listdir(_dir) if f.endswith('.csv')] 

    dataframes = []

    for file in files:
        file_path = os.path.json(_dir, file) 
        df = pd.read_csv(file_path)
        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)
    
      

if __name__=="__main__":
    process()