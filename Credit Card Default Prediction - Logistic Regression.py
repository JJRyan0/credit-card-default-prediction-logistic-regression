
# coding: utf-8

# **Python Sci-kit Learn: Machine Learning: Logistic Regression - Predicting Next Months Credit Card Default Payments**
# 
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">**Note:** Citation:  Data Source:Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.. Datasource@ http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients</div>
# 
# 
# Created by John Ryan 26th April 2017
# 
# 
# 
# __Overview__
# 
# Question: How do we predict future Credit card default payments with the measurements provided?
# 
# The data set provides many categorical and continous variables that allow for the opportunity to implement Machine Learning models to accurately predict future tranactional habits of customers.

# **Process of cleansing and transforming datasets to create a comprehensive data set which captures all possible hypothesis abd produces accurate predictive Models**
# 
# **Data Exploration and Preperation**
# 
# 1.Variable Identification
# 
# - Identify ID, Input and Target features
# 
# - Identify categorical and numerical features
# 
# - Identify columns with missing values
# 
# 2.Univariate Analysis
# 
# 3.Bi-variate Analysis
# 
# 4.Missing values treatment
# 
# 5.Outlier treatment
# 
# **Feature-Engineering**
# 
# 6.Variable transformation
# 
#  - Logarithm
#         
#  - Square/Cube root
#         
#  - Binning
#  
#  - Seasonality effects due to periods
#  
#  5.Scaling the data i.e standard, min-max
#         
#  Note: decision tree models generally do not require features to be
#  standardized or normalized, nor do they require categorical features to be binary-encoded.
#         
#  6.Dimensionality reduction does not focus on making predictions. Instead,
#   it tries to take a set of input data with a feature dimension D.
#   
#   - Principal Components Analysis (PCA)
#             
#   - Singular Value Decomposition (SVD)
#             
#   - Clustering
#    
#  7.Variable creation
#    
# Iterate over last 4 until to refine the model
# 
#  8.Feature -Importance
#  
# Note: Set list of Hypothesis in line with the problem you are trying to solve
# Select model for task at hand i.e classification, regression
# 
# 2. Data Modelling - Build Model
# 
#   - Encode Labels for categorical variables - One hot encoder.
#     
#   - Split data into Training and Test sets.
#     
#   - First Model Gradient Boosting Machine/Random Forest to create bencemark solution
#    create additional models.
#     
#   - K-fold cross Validation vs predicted score, record error rates
#    
# 3. Evaluate Model Performance Classification;
# 
#     - Confusion Matrices
#     - Accuracy and prediction error 
#     - Precision and Recall
#     - ROC curve and AUC
#     - The kappa statistic
#     - F-Measure
#     - KFold -Cross Validation
#     - Evaluate Model Performance Regression;
#       
# Multiple Linear Regression models: Understand how far away our predicted values are from
# true values, use metrics taking into account overall deviation; 
# 
# - Mean Squared Error
# - Root Mean Squared Logged Error 
# - Mean Absolute Error(MAE) 
# - R-Squared coefficient
#     
# 4 Optimization- Model selection and parameter tuning pipelines
# 
# - Logistic Regression and SVM use Stochastic Gradient Decent(SGD)
# - Decision Trees - Tuning tree depth and impurity
# - Naive Bayes - Changing the lambda parameter for naïve Bayes
# 

# In[1]:

# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd
# We'll also import seaborn, a Python graphing library
get_ipython().magic(u'matplotlib inline')
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)


# In[2]:

#Import the dataset using csv file 
loandata = pd.read_csv("C:/data/creditcard.csv")


# In[3]:

#View the first 10 rows of the dataframe
loandata.head(10)


# In[4]:

loandata.describe()


# **Data Munging, Preprocessing**
# 
# **Checking for missing values in the data**

# In[5]:

#Data Munging
#Checking for missing values in the data
loandata.apply(lambda x: sum(x.isnull()),axis=0)


# In[6]:

#show the combined total number of missing values
print("\nTotal number of NAN missing values: {0}".format((loandata.shape[0] * loandata.shape[1]) - loandata.count().sum()))


# In[7]:

#Display of the total Education value counts
loandata['EDUCATION'].value_counts()


# In[8]:

#Fill in the NaN values with the most common education type in the data
loandata['EDUCATION'].fillna('university',inplace=True)


# In[9]:

#Marriage type counts
loandata['MARRIAGE'].value_counts()


# In[10]:

#Fill in the NaN values with the most common  type in the data
loandata['MARRIAGE'].fillna('single', inplace=True)


# In[11]:

#Checking for missing values in the data after missing value treatment
loandata.apply(lambda x: sum(x.isnull()),axis=0)


# **Value Counts of the target output variable**

# In[12]:

loandata['default payment next month'].value_counts()


# In[13]:

#A look at the target variable using a bar plot
import seaborn as sns
sns.countplot(x="default payment next month", data=loandata)
sns.plt.show()


# **Univariate Analysis - Continous Variables**
# 
# Analysis of central tendency and spread of the variables.

# **Step A: Central Tendency; Mean, Meadian, Mode, Min & Max**
# (out of scope for this example)

# **Step B: Measures of Dispersion; Range, Quartile, IQR, Variance, Standard Deviation, Skewness and Kurtosis**
# (out of scope for this example)

# **1. Violin, Swarm Plots**

# **2. Boxplots & Outlier Analysis**

# In[14]:

#Boxplot for applicant income sorted by Education type
loandata.boxplot(column='LIMIT_BAL', by = 'EDUCATION')
plt.show()


# In[15]:

#Boxplot for applicant Income sorted by Gender Type
loandata.boxplot(column='LIMIT_BAL', by = 'SEX')
plt.show()


# In[16]:

loandata.boxplot(column='LIMIT_BAL', by = 'MARRIAGE')
plt.show()


# **3.Histograms**

# In[21]:

#Boxplot for Loan Amounts Issued to Applicants
loandata.boxplot(column='LIMIT_BAL')
plt.show()


# In[17]:

#Limit Balance Histogram
loandata['LIMIT_BAL'].hist(bins = 50)
plt.show()


# **Transforming the data to treat outliers (Logging)**

# In[20]:

#import numpy library for the computation of np.log and create resulting histogram
import numpy as np
loandata['LIMIT_BAL_log'] = np.log(loandata['LIMIT_BAL'])
loandata['LIMIT_BAL_log'].hist(bins=20)


# In[23]:

#Boxplot for Loan balances logged results in less outliers and a more balanced distribution
loandata.boxplot(column='LIMIT_BAL_log')
plt.show()


# In[22]:

# One way we can extend this plot is adding a layer of individual points on top of
# it through Seaborn's striplot
# 
# We'll use jitter=True so that all the points don't fall in single vertical lines
# above the species
ax = sns.boxplot(x="EDUCATION", y="LIMIT_BAL", data=loandata)
ax = sns.stripplot(x="EDUCATION", y="LIMIT_BAL", data=loandata, jitter=True, edgecolor="gray");


# **Bi-variate Analysis - Continous Variables**

# In[24]:

loandata.plot(kind="scatter", x="BILL_AMT3", y="PAY_AMT3");


# **Univariate Analysis - Categorical Variables**
# 
# Frequency table distribution analysis;
# 

# In[30]:

#creating a pivot table to assess the Credit History of Applicants
#Step 1: Create a Frequency Table for Credit History, how many applicants Yes (0.1) vs. No (0.0)
freqtable = loandata['EDUCATION'].value_counts(ascending = True)
freqtable


# In[31]:

#Stacked Chart by Education Type
stack1 = pd.crosstab(loandata['EDUCATION'], loandata['default payment next month'])
stack1.plot(kind = 'bar', stacked=True, color=['red','green'], grid = False)
plt.show()


# In[32]:

#Data Munging
#Checking for missing values in the data
loandata.apply(lambda x: sum(x.isnull()),axis=0)


# In[33]:

#show number of missing values
print("\nNumber of NAN missing values : {0}".format((loandata.shape[0] * loandata.shape[1]) - loandata.count().sum()))


# **Building a predictive model: Logistic Regression**

# In[34]:

#Label encoder
from sklearn.preprocessing import LabelEncoder

for feature in loandata.columns:
    if loandata[feature].dtype=='object':
        le = LabelEncoder()
        loandata[feature] = le.fit_transform(loandata[feature])
loandata.tail(5)


# In[36]:

#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
    #Fit the model:
    model.fit(data[predictors],data[outcome])
  
    #Make predictions on training set:
    predictions = model.predict(data[predictors])
  
    #Print accuracy
    accuracy = metrics.accuracy_score(predictions,data[outcome])
    print "Accuracy : %s" % "{0:.3%}".format(accuracy)

    #Perform k-fold cross-validation with 5 folds
    kf = KFold(data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        train_predictors = (data[predictors].iloc[train,:])# Filter training data
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
    print "Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))

    #Fit the model again so that it can be refered outside the function:
    model.fit(data[predictors],data[outcome]) 


# **Create Logistic Regression Model**

# In[39]:

#Model 1: Logistic Regression Model - using Education
import numpy as np
outcome_var = 'default payment next month'
model = LogisticRegression()
predictor_var = ['EDUCATION']
classification_model(model,loandata, predictor_var, outcome_var)


# In[40]:

#Model 2: Logistic Regression - using additional variables
predictor_var = ['EDUCATION', 'SEX', 'MARRIAGE']
classification_model(model,loandata, predictor_var, outcome_var)


# Reference: https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/
