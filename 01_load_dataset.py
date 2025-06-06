import pandas as pd

df = pd.read_csv("Mall_Customers.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

