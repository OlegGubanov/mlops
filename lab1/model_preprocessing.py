from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys


file_path = sys.argv[1]
df = pd.read_csv(file_path)
columns = df.columns[0:-1]

scaler = StandardScaler()
df[columns] = scaler.fit_transform(df[columns])
df.to_csv(file_path, index=False)