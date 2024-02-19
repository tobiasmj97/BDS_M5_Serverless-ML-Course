# %% [markdown]
# # Iris Flower - Batch Prediction
# 
# 
# In this notebook we will, 
# 
# 1. Load the batch inference data that arrived in the last 24 hours
# 2. Predict the first Iris Flower found in the batch
# 3. Write the ouput png of the Iris flower predicted, to be displayed in Github Pages.

# %%
import pandas as pd
import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()

# %%
mr = project.get_model_registry()
model = mr.get_model("iris", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/iris_model.pkl")

# %% [markdown]
# We are downloading the 'raw' iris data. We explicitly do not want transformed data, reading for training. 
# 
# So, let's download the iris dataset, and preview some rows. 
# 
# Note, that it is 'tabular data'. There are 5 columns: 4 of them are "features", and the "variety" column is the **target** (what we are trying to predict using the 4 feature values in the target's row).

# %%
feature_view = fs.get_feature_view(name="iris", version=1)

# %% [markdown]
# Now we will do some **Batch Inference**. 
# 
# We will read all the input features that have arrived in the last 24 hours, and score them.

# %%
import datetime
from PIL import Image

batch_data = feature_view.get_batch_data()

y_pred = model.predict(batch_data)

y_pred

# %%
batch_data

# %% [markdown]
# Batch prediction output is the last entry in the batch - it is output as a file 'latest_iris.png'

# %%
flower = y_pred[y_pred.size-1]
flower_img = "assets/" + flower + ".png"
img = Image.open(flower_img)            

img.save("../../assets/latest_iris.png")

# %%
iris_fg = fs.get_feature_group(name="iris", version=1)
df = iris_fg.read()
df

# %%
label = df.iloc[-1]["variety"]
label

# %%
label_flower = "assets/" + label + ".png"

img = Image.open(label_flower)            

img.save("../../assets/actual_iris.png")

# %%
import pandas as pd

monitor_fg = fs.get_or_create_feature_group(name="iris_predictions",
                                  version=1,
                                  primary_key=["datetime"],
                                  description="Iris flower Prediction/Outcome Monitoring"
                                 )

# %%
from datetime import datetime
now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

data = {
    'prediction': [flower],
    'label': [label],
    'datetime': [now],
}
monitor_df = pd.DataFrame(data)
monitor_fg.insert(monitor_df)

# %%
history_df = monitor_fg.read()
history_df

# %%
#!pip install dataframe_image -q

# %%
import dataframe_image as dfi

df_recent = history_df.tail(5)
 
# If you exclude this image, you may have the same iris_latest.png and iris_actual.png files
# If no files have changed, the GH-action 'git commit/push' stage fails, failing your GH action (last step)
# This image, however, is always new, ensuring git commit/push will succeed.
dfi.export(df_recent, '../../assets/df_recent.png', table_conversion = 'matplotlib')

# %%
from sklearn.metrics import confusion_matrix

predictions = history_df[['prediction']]
labels = history_df[['label']]

results = confusion_matrix(labels, predictions)
print(results)

# %%
from matplotlib import pyplot
import seaborn as sns

# Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris flowers
if results.shape == (3,3):

    df_cm = pd.DataFrame(results, ['True Setosa', 'True Versicolor', 'True Virginica'],
                         ['Pred Setosa', 'Pred Versicolor', 'Pred Virginica'])

    cm = sns.heatmap(df_cm, annot=True)

    fig = cm.get_figure()
    fig.savefig("../../assets/confusion_matrix.png") 
    df_cm
else:
    print("Run the batch inference pipeline more times until you get 3 different iris flowers")    


