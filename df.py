import numpy as np
import pandas as pd

# NO DRIFT
df1 = pd.DataFrame({
    "age": np.random.normal(40, 5, 1000),
    "income": np.random.normal(600000, 50000, 1000)
})

df2 = pd.DataFrame({
    "age": np.random.normal(41, 5, 1000),
    "income": np.random.normal(610000, 50000, 1000)
})

df1.to_csv("train.csv", index=False)
df2.to_csv("new.csv", index=False)