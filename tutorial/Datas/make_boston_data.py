import pandas as pd
import numpy as np
from sklearn import datasets

boston = datasets.load_boston()

print(boston.data.shape, boston.target.shape)
columns = np.append(boston.feature_names, ['MEDV'])
df = pd.DataFrame(np.hstack([boston.data, boston.target[:, None]]), columns=columns)
df.to_csv("./boston_data.csv")
