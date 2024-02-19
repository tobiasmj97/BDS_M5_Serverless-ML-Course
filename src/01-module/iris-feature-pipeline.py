# %% [markdown]
# # Iris Flower - Feature Pipeline
# 
# In this notebook we will, 
# 
# 1. Run in either "Backfill" or "Normal" operation. 
# 2. IF *BACKFILL==True*, we will load our DataFrame with data from the iris.csv file 
# 
#    ELSE *BACKFILL==False*, we will load our DataFrame with one synthetic Iris Flower sample 
# 3. Write our DataFrame to a Feature Group

# %%
#!pip install -U hopsworks --quiet

# %% [markdown]
# Set **BACKFILL=True** if you want to create features from the iris.csv file containing historical data.

# %%
import random
import pandas as pd
import hopsworks

BACKFILL=False

# %% [markdown]
# ### Synthetic Data Functions
# 
# These synthetic data functions can be used to create a DataFrame containing a single Iris Flower sample.

# %%
def generate_flower(name, sepal_len_max, sepal_len_min, sepal_width_max, sepal_width_min, 
                    petal_len_max, petal_len_min, petal_width_max, petal_width_min):
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    df = pd.DataFrame({ "sepal_length": [random.uniform(sepal_len_max, sepal_len_min)],
                       "sepal_width": [random.uniform(sepal_width_max, sepal_width_min)],
                       "petal_length": [random.uniform(petal_len_max, petal_len_min)],
                       "petal_width": [random.uniform(petal_width_max, petal_width_min)]
                      })
    df['variety'] = name
    return df


def get_random_iris_flower():
    """
    Returns a DataFrame containing one random iris flower
    """
    virginica_df = generate_flower("Virginica", 8, 5.5, 3.8, 2.2, 7, 4.5, 2.5, 1.4)
    versicolor_df = generate_flower("Versicolor", 7.5, 4.5, 3.5, 2.1, 3.1, 5.5, 1.8, 1.0)
    setosa_df =  generate_flower("Setosa", 6, 4.5, 4.5, 2.3, 1.2, 2, 0.7, 0.3)

    # randomly pick one of these 3 and write it to the featurestore
    pick_random = random.uniform(0,3)
    if pick_random >= 2:
        iris_df = virginica_df
    elif pick_random >= 1:
        iris_df = versicolor_df
    else:
        iris_df = setosa_df

    return iris_df

# %% [markdown]
# ## Backfill or create new synthetic input data
# 
# You can run this pipeline in either *backfill* or *synthetic-data* mode.

# %%

if BACKFILL == True:
    iris_df = pd.read_csv("https://repo.hops.works/master/hopsworks-tutorials/data/iris.csv")
else:
    iris_df = get_random_iris_flower()
    
iris_df.head()

# %% [markdown]
# ## Authenticate with Hopsworks using your API Key
# 
# Hopsworks will prompt you to paste in your API key and provide you with a link to find your API key if you have not stored it securely already.

# %%
project = hopsworks.login()
fs = project.get_feature_store()

# %% [markdown]
# ## Create and write to a feature group - primary keys
# 
# To prevent duplicate entries, Hopsworks requires that each DataFame has a *primary_key*. 
# A *primary_key* is one or more columns that uniquely identify the row. Here, we assume
# that each Iris flower has a unique combination of ("sepal_length","sepal_width","petal_length","petal_width")
# feature values. If you randomly generate a sample that already exists in the feature group, the insert operation will fail.
# 
# The *feature group* will create its online schema using the schema of the Pandas DataFame.

# %%
iris_fg = fs.get_or_create_feature_group(name="iris",
                                  version=1,
                                  primary_key=["sepal_length","sepal_width","petal_length","petal_width"],
                                  description="Iris flower dataset"
                                 )
iris_fg.insert(iris_df)

# %%



