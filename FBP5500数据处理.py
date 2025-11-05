import pandas as pd
import numpy as np
import os

data = pd.read_excel("SCUT-FBP5500_v2/SCUT-FBP5500_v2/All_Ratings.xlsx",sheet_name="ALL")
df = pd.DataFrame(data)

for i in range(len(df)):
    if pd.isna(df.loc[i,'original Rating']):
        df.loc[i,'original Rating']=df.loc[i,'Rating']
    df.loc[i,"averRating"]=(df.loc[i,'original Rating']+df.loc[i,'Rating'])/2
Facescore = df.groupby('Filename')['averRating'].mean().reset_index()
Facescore.columns= ['Filename','labels']
Facescore.to_csv('Facescore.csv',index=False)
print(Facescore)
