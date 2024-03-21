# %% [markdown]
# 

# %% [markdown]
# # Create Feature Groups and Backfill Features
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/featurestoreorg/serverless-ml-course/blob/main/src/02-module/1_backfill_cc_feature_groups.ipynb)
# 
# **Note**: you may get an error when installing hopsworks on Colab, and it is safe to ignore it.
# 
# 
# First, you need to install the hopsworks library.

# %%
#!pip install -U hopsworks --quiet

# %% [markdown]
# ## <span style="color:#ff5f27;"> 💽 Loading the Data </span>
# 
# The backfill data you will use comes from three different CSV files:
# 
# - `credit_cards.csv`: credit card information such as expiration date and provider.
# - `transactions.csv`: transaction information such as timestamp, location, and the amount. Importantly, the binary `fraud_label` variable tells us whether a transaction was fraudulent or not.
# - `profiles.csv`: credit card user information such as birthdate and city of residence.
# 
# You can conceptualize these CSV files as originating from separate data sources.
# **All three files have a credit card number column `cc_num` in common (a natural join key).**
# 

# %%
import pandas as pd
from datetime import datetime
import hopsworks

# %%
url = "https://repo.hops.works/master/hopsworks-tutorials/data/card_fraud_data"
credit_cards_df = pd.read_parquet(url + "/credit_cards.parquet")
credit_cards_df.head(5)

# %%
credit_cards_df.info()

# %%
profiles_df = pd.read_parquet(url + "/profiles.parquet")
profiles_df.head(5)

# %%
profiles_df.info()

# %%
trans_df = pd.read_parquet(url + "/transactions.parquet")
trans_df.head(3)

# %%
trans_df.info()

# %% [markdown]
# ## <span style="color:#ff5f27;"> 🛠️ Feature Engineering </span>
# 
# Fraudulent transactions can differ from regular ones in many different ways. Typical red flags would for instance be a large transaction volume/frequency in the span of a few hours. It could also be the case that elderly people in particular are targeted by fraudsters. To facilitate model learning you will create additional features based on these patterns. In particular, you will create two types of features:
# 1. **Features that aggregate data from different data sources**. This could for instance be the age of a customer at the time of a transaction, which combines the `birthdate` feature from `profiles.csv` with the `datetime` feature from `transactions.csv`.
# 2. **Features that aggregate data from multiple time steps**. An example of this could be the transaction frequency of a credit card in the span of a few hours, which is computed using a window function.
# 
# Let's start with the first category.

# %%
from sml import cc_features
import warnings
warnings.filterwarnings('ignore')

fraud_labels = trans_df[["tid", "cc_num", "datetime", "fraud_label"]]
fraud_labels.datetime = fraud_labels.datetime.map(lambda x: cc_features.date_to_timestamp(x))

trans_df = trans_df.drop(['fraud_label'], axis=1)
trans_df = cc_features.card_owner_age(trans_df, profiles_df)
trans_df = cc_features.expiry_days(trans_df, credit_cards_df)
trans_df = cc_features.activity_level(trans_df, 1)

trans_df

# %% [markdown]
# Next, you will create features that for each credit card aggregate data from multiple time steps.
# 
# Yoy will start by computing the distance between consecutive transactions, lets call it `loc_delta`.
# Here you will use the [Haversine distance](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html?highlight=haversine#sklearn.metrics.pairwise.haversine_distances) to quantify the distance between two longitude and latitude coordinates.

# %% [markdown]
# Before you can create a feature group you need to connect to Hopsworks feature store.

# %%
project = hopsworks.login()
fs = project.get_feature_store()

# %% [markdown]
# Next lets compute windowed aggregates. Here you will use 4-hour windows, but feel free to experiment with different window lengths by setting `window_len` below to a value of your choice.

# %%
window_len = 4
window_aggs_df = cc_features.aggregate_activity_by_hour(trans_df, window_len)
window_aggs_df.tail()

# %% [markdown]
# ## <span style="color:#ff5f27;"> 🪄 Creating Feature Groups </span>
# 
# A [feature group](https://docs.hopsworks.ai/feature-store-api/latest/generated/feature_group/) can be seen as a collection of conceptually related features. In this case, you will create a feature group for the transaction data and a feature group for the windowed aggregations on the transaction data. Both will have `cc_num` as primary key, which will allow you to join them when creating a dataset in the next tutorial.
# 
# Feature groups can also be used to define a namespace for features. For instance, in a real-life setting you would likely want to experiment with different window lengths. In that case, you can create feature groups with identical schema for each window length. 

# %% [markdown]
# To create a feature group you need to give it a name and specify a primary key. It is also good to provide a description of the contents of the feature group and a version number, if it is not defined it will automatically be incremented to `1`.

# %%
trans_fg = fs.get_or_create_feature_group(
    name="cc_trans_fraud",
    version=2,
    description="Credit Card transactions",
    primary_key=["cc_num"],
    event_time="datetime"
)

# %% [markdown]
# A full list of arguments can be found in the [documentation](https://docs.hopsworks.ai/feature-store-api/latest/generated/api/feature_store_api/#create_feature_group).
# 
# At this point, you have only specified some metadata for the feature group. It does not store any data or even have a schema defined for the data. To make the feature group persistent you need to populate it with its associated data using the `insert` function.

# %%
trans_fg.insert(trans_df, write_options={"wait_for_job" : False})

# %%
feature_descriptions = [
    {"name": "tid", "description": "Transaction id"},
    {"name": "datetime", "description": "Transaction time"},
    {"name": "cc_num", "description": "Number of the credit card performing the transaction"},
    {"name": "category", "description": "Expense category"},
    {"name": "amount", "description": "Dollar amount of the transaction"},
    {"name": "city", "description": "City in which the transaction was made"},
    {"name": "country", "description": "Country in which the transaction was made"},
    {"name": "age_at_transaction", "description": "Age of the card holder when the transaction was made"},
    {"name": "days_until_card_expires", "description": "Card validity days left when the transaction was made"},
    {"name": "loc_delta_t_minus_1", "description": "Haversine distance between this transaction location and the previous transaction location from the same card"},
    {"name": "time_delta_t_minus_1", "description": "Time in days between this transaction and the previous transaction location from the same card"},
]

for desc in feature_descriptions: 
    trans_fg.update_feature_description(desc["name"], desc["description"])

# %% [markdown]
# At the creation of the feature group, you will be prompted with an URL that will directly link to it; there you will be able to explore some of the aspects of your newly created feature group.
# 
# [//]: <> (insert GIF here)

# %% [markdown]
# You can move on and do the same thing for the feature group with our windows aggregation.

# %%
window_aggs_fg = fs.get_or_create_feature_group(
    name=f"cc_trans_fraud_{window_len}h",
    version=2,
    description=f"Counts of the number of credit card transactions over {window_len} hour windows.",
    primary_key=["cc_num"],
    event_time="datetime"
)

# %%
window_aggs_fg.insert(window_aggs_df, write_options={"wait_for_job" : False})

# %%
feature_descriptions = [
    {"name": "datetime", "description": "Transaction time"},
    {"name": "cc_num", "description": "Number of the credit card performing the transaction"},
    {"name": "loc_delta_mavg", "description": "Moving average of location difference between consecutive transactions from the same card"},
    {"name": "trans_freq", "description": "Moving average of transaction frequency from the same card"},
    {"name": "trans_volume_mavg", "description": "Moving average of transaction volume from the same card"},
    {"name": "trans_volume_mstd", "description": "Moving standard deviation of transaction volume from the same card"},
]

for desc in feature_descriptions: 
    window_aggs_fg.update_feature_description(desc["name"], desc["description"])

# %%
trans_label_fg = fs.get_or_create_feature_group(
    name="transactions_fraud_label",
    version=2,
    description="CC transactions that have been flagged as fraud",
    primary_key=['cc_num'],
    event_time='datetime'
)

trans_label_fg.insert(fraud_labels, write_options={"wait_for_job" : False})

# %%
feature_descriptions = [
    {"name": "tid", "description": "Transaction id"},
    {"name": "cc_num", "description": "Number of the credit card performing the transaction"},    
    {"name": "datetime", "description": "Transaction time"},
    {"name": "fraud_label", "description": "Whether the transaction was fraudulent or not"},
]
for desc in feature_descriptions: 
    trans_label_fg.update_feature_description(desc["name"], desc["description"])

# %% [markdown]
# Both feature groups are now accessible and searchable in the UI

# %% [markdown]
# ## <span style="color:#ff5f27;">⏭️ **Next:** Synthetic Data Feature Pipeline </span>
# 
# In the following notebook you will use create synthetic data that will be used to create features that are written to the  feature groups you created here.

# %%



