import pandas as pd

url = "https://archive.ics.uci.edu/static/public/880/data.csv"
df = pd.read_csv(url)
df.to_csv("data/data.csv", index=False)
print(f"Saved! Shape: {df.shape}")