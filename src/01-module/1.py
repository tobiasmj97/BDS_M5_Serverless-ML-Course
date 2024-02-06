# %% [markdown]
# # Iris Flower Train and Publish Model
# 
# 
# In this notebook we will, 
# 
# 1. Load the Iris Flower dataset into random split (train/test) DataFrames using a Feature View
# 2. Train a KNN Model using SkLearn
# 3. Evaluate model performance on the test set
# 4. Register the model with Hopsworks Model Registry

# %%
#pip install -U hopsworks --quiet

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import hopsworks

# %% [markdown]
# Let's first get a feature_view for the iris flower dataset, or create one if it does not already exist.
# If you are running this notebook for the first time, it will create the feature view, which contains all of the columns from the **iris feature group**.
# 
# There are 5 columns: 4 of them are "features", and the **variety** column is the **label** (what we are trying to predict using the 4 feature values in the label's row). The label is often called the **target**.

# %%
project = hopsworks.login()
fs = project.get_feature_store()

try: 
    feature_view = fs.get_feature_view(name="iris", version=1)
except:
    iris_fg = fs.get_feature_group(name="iris", version=1)
    query = iris_fg.select_all()
    feature_view = fs.create_feature_view(name="iris",
                                      version=1,
                                      description="Read from Iris flower dataset",
                                      labels=["variety"],
                                      query=query)

# %% [markdown]
# We will read our features and labels split into a **train_set** and a **test_set**. You split your data into a train_set and a test_set, because you want to train your model on only the train_set, and then evaluate its performance on data that was not seen during training, the test_set. This technique helps evaluate the ability of your model to accurately predict on data it has not seen before.
# 
# We can ask the feature_view to return a **train_test_split** and it returns:
# 
# * **X_** is a vector of features, so **X_train** is a vector of features from the **train_set**. 
# * **y_** is a scale of labels, so **y_train** is a scalar of labels from the **train_set**. 
# 
# Note: a vector is an array of values and a scalar is a single value.
# 
# Note: that mathematical convention is that a vector is denoted by an uppercase letter (hence "X") and a scalar is denoted by a lowercase letter (hence "y").
# 
# **X_test** is the features and **y_test** is the labels from our holdout **test_set**. The **test_set** is used to evaluate model performance after the model has been trained.

# %%
X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)

# %%
y_train

# %% [markdown]
# Now, we can fit a model to our features and labels from our training set (**X_train** and **y_train**). 
# 
# Fitting a model to a dataset is more commonly called "training a model".

# %%
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, y_train.values.ravel())

# %% [markdown]
# Now, we have trained our model. We can evaluate our model on the **test_set** to estimate its performance.

# %%
y_pred = model.predict(X_test)
y_pred

# %% [markdown]
# We can report on how accurate these predictions (**y_pred**) are compared to the labels (the actual results - **y_test**). 

# %%
from sklearn.metrics import classification_report

metrics = classification_report(y_test, y_pred, output_dict=True)
print(metrics)

# %%
from sklearn.metrics import confusion_matrix

results = confusion_matrix(y_test, y_pred)
print(results)

# %% [markdown]
# Notice in the confusion matrix results that we have 1 or 2 incorrect predictions.
# We have only 30 flowers in our test set - **y_test**.
# Our model predicted 1 or 2 flowers were of type "Virginica", but the flowers were, in fact, "Versicolor".

# %%
from matplotlib import pyplot

df_cm = pd.DataFrame(results, ['True Setosa', 'True Versicolor', 'True Virginica'],
                     ['Pred Setosa', 'Pred Versicolor', 'Pred Virginica'])

cm = sns.heatmap(df_cm, annot=True)

fig = cm.get_figure()
fig.savefig("assets/confusion_matrix.png") 
fig.show()

# %% [markdown]
# ## Register the Model with Hopsworks Model Registry
# 
# 

# %%
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
import os
import joblib
import hopsworks
import shutil

project =  hopsworks.login()
mr = project.get_model_registry()

# The 'iris_model' directory will be saved to the model registry
model_dir="iris_model"
if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)
joblib.dump(model, model_dir + "/iris_model.pkl")
shutil.copyfile("assets/confusion_matrix.png", model_dir + "/confusion_matrix.png")

input_example = X_train.sample()
input_schema = Schema(X_train)
output_schema = Schema(y_train)
model_schema = ModelSchema(input_schema, output_schema)

iris_model = mr.python.create_model(
    version=1,
    name="iris", 
    metrics={"accuracy" : metrics['accuracy']},
    model_schema=model_schema,
    input_example=input_example, 
    description="Iris Flower Predictor")

iris_model.save(model_dir)

