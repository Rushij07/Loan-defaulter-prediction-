'''CRISP-ML(Q):
    1.a. Business problem: Significant proportion of customers are defaulting on loan
        i. Business Objectives: Minimize Loan Defaulters
        ii. Business Constraints: Maximize the Profits
        Success Criteria:
        i. Business success criteria: Reduce the loan defaulters by atleast 10%
        ii. ML success criteria: Achieve an accuracy of over 70%
        iii. Economic success criteria: Save the bank more than 1.2 MUSD because of reduction of loan defaulters
    1.b. Data Collection: Bank -> 1000 customers, 17 variables (16 Inputs and 1 Ouput) - 
    2. Data Preprocessing - Cleansing & EDA / Descriptive Analytics
    3. Model Building - Experiment with different models alongside Hyperparameter tuning
    4. Evaluation - Not just model evaluation based on accuracy but we also need to evaluate business & economic success criteria
    5. Model Deployment (Flask)
    6. Monitoring & Maintenance (Prediction results to the database - MS SQL)'''


# pip install sklearn_pandas
# pip install graphviz

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import joblib
import pickle



# Load the data into Python dataframe for bulk load to SQL
data = pd.read_csv(r"C:\Users\Rushikesh Jadhav\Downloads\DT_flask_final_v2\DT_flask_final_v2\credit.csv") 
data

data.columns

data.info()

# Checking for Null values
data.isnull().sum()

# ### AutoEDA
# sweetviz
##########

# pip install sweetviz
import sweetviz
my_report = sweetviz.analyze([data, "data"])

my_report.show_html('Report1.html')# D-Tale


data.columns

data = data.drop(["phone"], axis = 1) # Unwanted columns are removed.

# Target variable categories
data['default'].unique()

data['default'].value_counts()


# Data split into Input and Output
X = data.iloc[:, :15] # Predictors 

y = data['default'] # Target 


# #### Separating Numeric and Non-Numeric columns
numeric_features = X.select_dtypes(exclude = ['object']).columns
numeric_features

categorical_features = X.select_dtypes(include=['object']).columns
categorical_features


# ### Imputation to handle missing values 

# ### MinMaxScaler to convert the magnitude of the columns to a range of 0 to 1
num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean')),
                                 ('scale', MinMaxScaler())])


# ### Encoding - One Hot Encoder to convert Categorical data to Numeric values

## Encoding
# Categorical features
encoding_pipeline = Pipeline([('onehot', OneHotEncoder(drop = 'first'))])

# Creating a transformation of variable with ColumnTransformer()
preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features),
                                                 ('categorical', encoding_pipeline, categorical_features)])

imp_enc_scale = preprocessor.fit(X)

# #### Save the imputation model using joblib
joblib.dump(imp_enc_scale, 'imp_enc_scale', protocol=pickle.HIGHEST_PROTOCOL)

import os
os.getcwd()

cleandata = pd.DataFrame(imp_enc_scale.transform(X), 
                         columns = imp_enc_scale.get_feature_names_out())
cleandata

# if you get any error then update the scikit-learn library version & restart the kernel to fix it

# ### Outlier Analysis

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True

cleandata.iloc[:, 0:6].plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''
# Increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()

cleandata.iloc[:, 0:6].columns


# #### Outlier analysis: Columns 'months_loan_duration', 'amount', and 'age' are continuous, hence outliers are treated
winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = ['num__months_loan_duration', 'num__amount',
                                       'num__percent_of_income', 
                                       'num__years_at_residence',
                                       'num__age', 'num__existing_loans_count'])


outlier = winsor.fit(cleandata[['num__months_loan_duration', 'num__amount',
                                'num__percent_of_income', 'num__years_at_residence',
                                'num__age', 'num__existing_loans_count']])

# Save the winsorizer model 
joblib.dump(outlier, 'winsor')

cleandata[['num__months_loan_duration', 'num__amount',
           'num__percent_of_income',
           'num__years_at_residence',
           'num__age', 'num__existing_loans_count']] = outlier.transform(cleandata[['num__months_loan_duration',
                                                                                    'num__amount', 'num__percent_of_income', 
                                                                                    'num__years_at_residence', 'num__age',
                                                                                    'num__existing_loans_count']])
cleandata

cleandata.iloc[:, 0:6].plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


# Split data into train and test with Stratified sample technique
# from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(cleandata, y, test_size = 0.2, stratify = y, random_state = 0) 

# Proportion of Target variable categories are consistent across train and test
print(Y_train.value_counts()/ 800)
print("\n")
print(Y_test.value_counts()/ 200)


### Decision Tree Model
model = DT(criterion = 'entropy')
model.fit(X_train, Y_train)

# Prediction on Test Data
preds = model.predict(X_test)

preds


# Accuracy
print(accuracy_score(Y_test, preds))

pd.crosstab(Y_test, preds, rownames = ['Actual'], colnames = ['Predictions']) 


### Hyperparameter Optimization
# create a dictionary of all hyperparameters to be experimented
param_grid = { 'criterion':['gini', 'entropy'], 'max_depth': np.arange(3, 15)}

# Decision tree model
dtree_model = DT()

# GridsearchCV with cross-validation to perform experiments with parameters set
dtree_gscv = GridSearchCV(dtree_model, param_grid, cv = 5, scoring = 'accuracy', return_train_score = False, verbose = 1)


# Train
dtree_gscv.fit(X_train, Y_train)

# The best set of parameter values
dtree_gscv.best_params_


# Model with best parameter values
DT_best = dtree_gscv.best_estimator_
DT_best

# Prediction on Test Data
preds1 = DT_best.predict(X_test)
preds1

pd.crosstab(Y_test, preds, rownames = ['Actual'], colnames= ['Predictions']) 

# Accuracy
print(accuracy_score(Y_test, preds))


import os
import graphviz
from sklearn import tree
from sklearn.tree import export_graphviz


predictors = list(cleandata.columns)
type(predictors)

dot_data = tree.export_graphviz(DT_best, filled = True, rounded = True,
                                feature_names = predictors,
                                class_names = ['Default', "Not Default"],
                                out_file = None)

''' If graphviz library does not work then ensure that you download the zip file (that contains all tools and libraries) 
from the website "https://graphviz.org/download/" and then replace it in the anaconda path 
(see the path in the terminal while installing any library (e.g., pip install matplotlib)) and 
then run the lines of code - 265 and 266'''

import os
os.environ["PATH"] += os.pathsep + 'C:/Anaconda/Lib/site-packages/graphviz/bin'

graph = graphviz.Source(dot_data)
graph

# Prediction on Train Data
preds_train = DT_best.predict(X_train)
preds_train

# Confusion Matrix
pd.crosstab(Y_train, preds_train, rownames = ['Actual'], colnames = ['Predictions']) 


# Accuracy
print(accuracy_score(Y_train, preds_train))


# ### Save the Best Model with pickel library
pickle.dump(DT_best, open('DT.pkl', 'wb'))


## Model Training with Cross Validation
from sklearn.model_selection import cross_validate


def cross_validation(model, _X, _y, _cv=5):

    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator = model, X = _X, y = _y,
                           cv = _cv, scoring = _scoring,
                           return_train_score = True)

    return pd.DataFrame({"Training Accuracy scores": results['train_accuracy'],
          "Mean Training Accuracy": results['train_accuracy'].mean()*100,
          "Training Precision scores": results['train_precision'],
          "Mean Training Precision": results['train_precision'].mean(),
          "Training Recall scores": results['train_recall'],
          "Mean Training Recall": results['train_recall'].mean(),
          "Training F1 scores": results['train_f1'],
          "Mean Training F1 Score": results['train_f1'].mean(),
          "Validation Accuracy scores": results['test_accuracy'],
          "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
          "Validation Precision scores": results['test_precision'],
          "Mean Validation Precision": results['test_precision'].mean(),
          "Validation Recall scores": results['test_recall'],
          "Mean Validation Recall": results['test_recall'].mean(),
          "Validation F1 scores": results['test_f1'],
          "Mean Validation F1 Score": results['test_f1'].mean()
          })



# from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(Y_train)

'''label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))

print("Mapping of Label Encoded Classes", label_encoder_name_mapping, sep = "\n")
print("Label Encoded Target Variable", encoded_y, sep = "\n")'''


# from sklearn.tree import DecisionTreeClassifier
decision_tree_result = cross_validation(DT_best, X_train, encoded_y, 5)

decision_tree_result

def plot_result(x_label, y_label, plot_title, train_data, val_data):
       
        # Set size of plot
        plt.figure(figsize=(12, 6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis - 0.2, train_data, 0.4, color = 'blue', label = 'Training')
        plt.bar(X_axis + 0.2, val_data, 0.4, color = 'red', label = 'Validation')
        plt.title(plot_title, fontsize = 30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize = 14)
        plt.ylabel(y_label, fontsize = 14)
        plt.legend()
        plt.grid(True)
        plt.show()

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

model_name = "Decision Tree"
plot_result(model_name,
            "Accuracy",
            "Accuracy scores in 5 Folds",
            decision_tree_result["Training Accuracy scores"],
            decision_tree_result["Validation Accuracy scores"])

