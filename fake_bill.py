#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[ ]:


dataset = "Fake_Bills_Detection_Expanded.xlsx"
xls = pd.ExcelFile(dataset)


# In[ ]:


df = pd.read_excel(xls, sheet_name="fake_bills")
df


# In[4]:


df.info()


# In[5]:


df.head()


# In[6]:


duplicates = df.duplicated().sum()
duplicates


# In[7]:


summary_stats = df.describe()
summary_stats


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


plt.figure(figsize=(6, 4))
sns.boxplot(x=df['diagonal'])
plt.title("Boxplot of diagonal")
plt.grid(True)
plt.show()


# In[10]:


plt.figure(figsize=(6, 4))
sns.boxplot(x=df['height_left'])
plt.title("Boxplot of height_left")
plt.grid(True)
plt.show()


# In[11]:


plt.figure(figsize=(6, 4))
sns.boxplot(x=df['height_right'])
plt.title("Boxplot of height_right")
plt.grid(True)
plt.show()


# In[12]:


plt.figure(figsize=(6, 4))
sns.boxplot(x=df['margin_low'])
plt.title("Boxplot of margin_low")
plt.grid(True)
plt.show()


# In[13]:


plt.figure(figsize=(6, 4))
sns.boxplot(x=df['margin_up'])
plt.title("Boxplot of margin_up")
plt.grid(True)
plt.show()


# In[14]:


plt.figure(figsize=(6, 4))
sns.boxplot(x=df['length'])
plt.title("Boxplot of length")
plt.grid(True)
plt.show()


# In[15]:


plt.figure(figsize=(6, 4))
sns.countplot(x="is_genuine", data=df, palette=["blue", "red"])
plt.title("Distribution of Genuine vs Fake Bills", fontsize=14)
plt.xlabel("Is Genuine (1 = Real, 0 = Fake)")
plt.ylabel("Count")
plt.grid(axis="y")
plt.show()

corr_matrix = df.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.show()


# In[16]:


df.fillna(df.mean(), inplace=True)  
df.info()


# #### Observations
# - There are 100000 rows and 7 columns
# - there are no null values
# 

# In[18]:


duplicate_count = df.duplicated().sum()
print(f"Total Duplicate Rows: {duplicate_count}")


# #### Observations ####
# - there are no duplicate rows
# 

# In[20]:


counts = df["is_genuine"].value_counts()
sns.barplot(data = counts)


# In[21]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df.iloc[:, 0] = labelencoder.fit_transform(df.iloc[:,0])
df.head()


# In[22]:


df.info()


# #### Observations ####
# - now the datatype of the is_genuine is changed for boolean to int

# In[24]:


df.head(3)


# In[25]:


X = df.iloc[:,1:6]
Y = df['is_genuine']


# In[26]:


Y


# In[27]:


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3,random_state = 1)
x_train


# In[28]:


from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# In[29]:


from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Define feature matrix and target vector
X = df.iloc[:, 1:]
Y = df.iloc[:, 0]

# Reduce the number of splits for faster cross-validation
kfold = StratifiedKFold(n_splits=3, random_state=2023, shuffle=True)

# Random Forest Classifier with parallel processing enabled
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# Optimize parameter space to reduce computation
params = {
    'max_depth': [3, 5, None],  # Remove depth 2 to make better splits and reduce redundancy
    'min_samples_leaf': [10, 20],  # Remove smaller leaves to minimize overfitting
    'n_estimators': [100, 200],  # Reduce the number of estimators to balance speed and accuracy
    'max_features': ["sqrt", "log2"],  # Remove 'None' for better feature selection efficiency
    'criterion': ["gini"]  # Focus on "gini" for faster computation
}

# Perform hyperparameter tuning with reduced iterations
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=params,
    n_iter=10,  # Lowered iterations for faster execution
    cv=kfold,
    n_jobs=-1,
    verbose=10,
    scoring="accuracy",
    random_state=42
)

# Fit the model
random_search.fit(X, Y)

print("Best Parameters:", random_search.best_params_)
print("Best Accuracy Score:", random_search.best_score_)


# In[84]:


random_search.best_estimator_


# In[ ]:





# #### Feature selection using RandomForest

# In[90]:


model_best=RandomForestClassifier(criterion='entropy', max_depth=5, max_features=None,
                       min_samples_leaf=5, n_jobs=-1, random_state=42)
model_best.fit(X,Y)
model_best.feature_importances_


# In[98]:


df=pd.DataFrame(model_best.feature_importances_,columns=["Importance score"],index=X.columns)
df.sort_values(by="Importance score")


# In[100]:


model_best.fit(x_train, y_train)
y_pred = model_best.predict(x_test)
print(classification_report(y_test, y_pred))


# #### XGBOOST

# In[56]:


pip install xgboost


# In[60]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler


# In[102]:


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled)
print(x_test_scaled)


# In[106]:


xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 150, 200, 300],
    'learning_rate': [0.01, 0.1, 0.15],
    'max_depth': [2, 3, 4, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Define the Stratified K-Fold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform Randomized Search
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,  # Corrected from param_dist to param_grid
    n_iter=10,
    cv=skf,
    scoring='recall',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Fit the model on training data
random_search.fit(x_train, y_train)

# Output the results
print("Best Parameters:", random_search.best_params_)
print("Best Recall Score:", random_search.best_score_)


# In[107]:


random_search.fit(x_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best Recall Score:", random_search.best_score_)


# In[108]:


best_model = random_search.best_estimator_

y_pred = best_model.predict(x_test_scaled)


# In[109]:


print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[110]:


best_model.feature_importances_


# In[111]:


import pandas as pd

features = pd.DataFrame(best_model.feature_importances_, 
                        index=x_train.columns, 
                        columns=["Importances"])

df1 = features.sort_values(by="Importances", ascending=False)

print(df1)


# In[121]:


import pandas as pd

# Extract feature importances from the best model
features = pd.DataFrame(
    best_model.feature_importances_,
    index=x_train.columns,  # Use X_train columns to match features
    columns=["Importance"]
)

# Sort feature importances in descending order
df1 = features.sort_values(by="Importance", ascending=False)

# Display the feature importances
print(df1)


# In[123]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plot feature importances
plt.figure(figsize=(10, 6))  # Set figure size
sns.barplot(data=df1, x=df1.index, y="Importance", palette="Set2")

# Improve readability
plt.xticks(rotation=90)  # Rotate feature names for clarity
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importances")

# Show plot
plt.show()



# In[125]:


# Example input data for prediction (replace this with your test data)
input_data = x_test  # Assuming x_test is your test dataset

# Predict the labels for the input data
predictions = model_best.predict(input_data)

# Output the predictions as an array
print("Predictions:", predictions)


# In[ ]:





# In[ ]:




