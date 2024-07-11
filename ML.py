#(1)linear regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import statsmodels.api as sm

#input dataset
NK_AML=pd.DataFrame(pd.read_csv(r"D:/Python/(6) machine learning/AML_nk_80_ID.csv"))

#check the column names
NK_AML_02=pd.DataFrame(NK_AML.columns.values.tolist())

# established the linear model
NK_AML_03=NK_AML.dropna(subset=['AGE_01'])
x = NK_AML_03[['ESTIMATE_score','Stromal_score','Immune_score']]
y = NK_AML_03['AGE_01']

# Add constant to the predictor variables
x = sm.add_constant(x)

# Create and fit the linear regression model
model_regr = sm.OLS(y, x)
results = model_regr.fit()

# Print the summary statistics
print(results.summary())


#(2)logistic regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score


#input dataset
NK_AML=pd.DataFrame(pd.read_csv(r"D:/Python/(6) machine learning/AML_nk_80_ID.csv"))

# Prepare the feature matrix X and the target variable y
NK_AML_01=NK_AML.dropna(subset=['AGE_01'])
X = NK_AML_01[['ESTIMATE_score','Stromal_score','Immune_score']]  
y = NK_AML_01['AGE_02'] 

# Split the data into training and testing sets (test_size, testing for 20% sample from original dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Create and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
probabilities = model.predict_proba(X_test)[:, 1]
auc_roc = roc_auc_score(y_test, probabilities)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("AUC-ROC:", auc_roc)

#Q-learning

#R-learning

#TD-learning
