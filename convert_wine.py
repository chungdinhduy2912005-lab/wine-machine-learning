import pandas as pd

# Define column names based on wine.names
# 1st column is Class (target)
columns = [
    "class",
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315",
    "proline"
]

# Read raw data
df = pd.read_csv("data/wine.data", names=columns)

# Save to CSV
df.to_csv("data/wine.csv", index=False)
print("Convert wine.data to data/wine.csv successfully. Shape:", df.shape)
