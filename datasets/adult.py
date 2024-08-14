# https://github.com/itdxer/adult-dataset-analysis/blob/master/Data%20analysis.ipynb

from collections import OrderedDict
import pandas as pd
import numpy as np

data_types = OrderedDict([
    ("age", "int"),
    ("workclass", "category"),
    ("final-weight", "int"),  # originally it was called fnlwgt
    ("education", "category"),
    ("education-num", "int"),
    ("marital-status", "category"),
    ("occupation", "category"),
    ("relationship", "category"),
    ("race", "category"),
    ("sex", "category"),
    ("capital-gain", "float"),  # required because of NaN values
    ("capital-loss", "int"),
    ("hours-per-week", "int"),
    ("native-country", "category"),
    ("salary-class", "category"),
])
target_column = "income_class"

path = "adult/adult.data"

df = pd.read_csv(
    path,
    names=data_types,
    index_col=None,
    dtype=data_types,
    sep=", "
)

# attributes = ["age", "workclass", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "hours-per-week", "native-country", "salary-class"]
# df = df[attributes]

df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

print(df.shape)
df.to_csv("adult.csv", index=False)

df[['sex',
 'age',
 'race',
 'marital-status',
 'education',
 'native-country',
 'workclass',
 'occupation',
 'salary-class']].to_csv("adult_orig.csv", index=False)
