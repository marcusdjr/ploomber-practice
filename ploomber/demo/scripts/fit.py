# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Add description here
#
# *Note:* You can open this file as a notebook (JupyterLab: right-click on it in the side bar -> Open With -> Notebook)


# %%
# Uncomment the next two lines to enable auto reloading for imported modules
# # %load_ext autoreload
# # %autoreload 2
# For more info, see:
# https://docs.ploomber.io/en/latest/user-guide/faq_index.html#auto-reloading-code-in-jupyter

# %% tags=["parameters"]
# If this task has dependencies, list them them here
# (e.g. upstream = ['some_task']), otherwise leave as None.
upstream = ['get', 'sepal-feature', 'petal-feature']

# This is a placeholder, leave it as None
product = None


# %%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn_evaluation.plot import confusion_matrix

from pathlib import Path
import pickle

# %%
raw = pd.read_csv(upstream['get']['data'])

# %%
raw

# %%
sepal = pd.read_csv(upstream['sepal-feature']['data'])

# %%
petal = pd.read_csv(upstream['petal-feature']['data'])

# %%
df = raw.join(sepal).join(petal)

# %%
df

# %%
X = df.drop('target', axis='columns')
y = df.target

# %%
model = RandomForestClassifier()

# %%
model.fit(X,y)

# %%
y_pred = model.predict(X)

# %%
confusion_matrix(y,y_pred)

# %%
Path(product['model']).write_bytes(pickle.dumps(model))

# %%
