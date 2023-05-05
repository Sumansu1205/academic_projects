from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
import pandas as pd
y_test = [0, 1, 0, 0, 1, 0]
y_pred = [0, 1, 1, 1, 0, 0]

# 0: Customers that made payment on time
# 1: Customers who defaulted

print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))

print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=[
      'Actual:0', 'Actual:1'], columns=['Pred:0', 'Pred:1']))

print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))

print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=[
      'Actual:0', 'Actual:1'], columns=['Pred:0', 'Pred:1']))

#Part 2-****************************************************************************
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('/Users/surabhisuman/Downloads/default-2-1.csv')
print(df)

# EDA

sns.boxplot(x='default', y='balance', data=df)
plt.show()

sns.boxplot(x='default', y='income', data=df)
plt.show()


df = pd.get_dummies(df, drop_first=True)
print(df)

x = df[['balance', 'income', 'student_Yes']]
y = df[['default_Yes']]

# train test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=101)

# ''Demonstration Only''
# from sklearn.linear_model import LinearRegression
# lr=LinearRegression()
# lr.fit(x_train, y_train)

# y_pred=lr.predict(x_test)
# print(y_pred)

# #Thniking of y_pred as probability of default
# #If probability greater than 0.5>> classify as 1
# #If less than 0.5>> classify as 0

# plt.scatter(x_test, y_pred)
# plt.xlabel('Balance')
# plt.ylabel('Predicted probability of default')
# plt.show()

# Implementing logistic regression model***********

logmodel = LogisticRegression(solver='liblinear')  # import initialise train
logmodel.fit(x_train, y_train)  # fit

logmodel.intercept_
logmodel.coef_

probabilities = logmodel.predict_proba(x_test)

y_pred = logmodel.predict(x_test)
print(y_pred)

# plt.scatter(x_test, probabilities[:,1])
# plt.scatter(x_test, y_pred, label='Predicted Values')
# plt.xlabel('Balance')
# plt.ylabel('Predicted Probabailities of default')
# plt.legend()
# plt.show()

# Evaluate model using F1 score
print(f1_score(y_test, y_pred))

# x1=df[['student', 'balance', 'income']]
# y1=df[['default_Yes']]

# #try to use other variables
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train,y_test=train_test_split(x1,y1,test_size=0.3, random_state=101)

# logmodel = LogisticRegression(solver='liblinear') #import initialise train
# logmodel.fit(x_train,y_train) #fit

# logmodel.intercept_
# logmodel.coef_

# probabilities = logmodel.predict_proba(x_test)

# y_pred= logmodel.predict(x_test)
# print(y_pred)

# plt.scatter(x_test, probabilities[:,1])
# plt.scatter(x_test, y_pred, label='Predicted Values')
# plt.xlabel('Balance')
# plt.ylabel('Predicted Probabailities of default')
# plt.legend()
# plt.show()

# print(f1_score(y_test,y_pred))


#Part 3:-    ********************************************************
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


df = pd.read_csv('/Users/surabhisuman/Downloads/eBayAuctions.csv')

'''Get the data, Deal with categorical data by creating dummy variables and remove
the rows that
have missing values'''

df = pd.get_dummies(df, drop_first=True)
print(df)

print(df.isna().sum())

'''Split data into training (70%) and testing (30%) data. Build a logistic
regression
model on the training data and evaluate the F1-score on the test data
Use random state=1'''

x = df[['sellerRating', 'Duration', 'ClosePrice', 'OpenPrice', 'Category_Automotive', 'Category_Books', 'Category_Business/Industrial', 'Category_Clothing/Accessories', 'Category_Coins/Stamps', 'Category_Collectibles', 'Category_Computer', 'Category_Electronics', 'Category_EverythingElse', 'Category_Health/Beauty', 'Category_Home/Garden',
        'Category_Jewelry', 'Category_Music/Movie/Game', 'Category_Photography', 'Category_Pottery/Glass', 'Category_SportingGoods', 'Category_Toys/Hobbies', 'currency_GBP', 'currency_US', 'endDay_Mon', 'endDay_Sat', 'endDay_Sun', 'endDay_Thu', 'endDay_Tue', 'endDay_Wed']]
y = df['Competitive?']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1)
logmodel = LogisticRegression(solver='liblinear')  # import initialise train
logmodel.fit(x_train, y_train)  # fit

logmodel.intercept_
logmodel.coef_

probabilities = logmodel.predict_proba(x_test)

y_pred = logmodel.predict(x_test)
print(y_pred)
print(f1_score(y_test, y_pred))  # 0.78418

'''If we want to predict at the start of an auction whether it will be competitive,
we cannot
use the information on the closing price. Run a logistic model with all predictors
as above,
excluding price. How does this model compare to the full model with respect to the
previous model
based on the f1-score?'''

x1 = df[['sellerRating', 'Duration', 'OpenPrice', 'Category_Automotive', 'Category_Books', 'Category_Business/Industrial', 'Category_Clothing/Accessories', 'Category_Coins/Stamps', 'Category_Collectibles', 'Category_Computer', 'Category_Electronics', 'Category_EverythingElse', 'Category_Health/Beauty', 'Category_Home/Garden',
         'Category_Jewelry', 'Category_Music/Movie/Game', 'Category_Photography', 'Category_Pottery/Glass', 'Category_SportingGoods', 'Category_Toys/Hobbies', 'currency_GBP', 'currency_US', 'endDay_Mon', 'endDay_Sat', 'endDay_Sun', 'endDay_Thu', 'endDay_Tue', 'endDay_Wed']]
y1 = df['Competitive?']


x_train, x_test, y_train, y_test = train_test_split(
    x1, y1, test_size=0.3, random_state=1)
logmodel = LogisticRegression(solver='liblinear')  # import initialise train
logmodel.fit(x_train, y_train)  # fit

logmodel.fit(x_train, y_train)  # fit

logmodel.intercept_
logmodel.coef_

probabilities = logmodel.predict_proba(x_test)

y_pred = logmodel.predict(x_test)
print(y_pred)
f1_without_closing = f1_score(y_test, y_pred)  # 0.68389
# print(df.shape)

'''Use forward selection to improve the model in the previous step, use f1_score as
the metric.
Does the model improve??'''
len(x_train.columns)
sfs = SFS(logmodel,
          k_features=(1, 28),
          forward=True,
          scoring='f1',
          cv=20)

sfs.fit(x_train, y_train)

# what features were selected
print(sfs.k_feature_names_)  # ('sellerRating', 'Category_Business/Industrial', 'Category_Coins/Stamps', 'Category_Electronics', 'Category_Health/Beauty', 'Category_SportingGoods', 'Category_Toys/Hobbies')
X_train_sfs = sfs.transform(x_train)
X_test_sfs = sfs.transform(x_test)

# Fit the model using the new feature subset
# and make a prediction on the test data
logmodel.fit(X_train_sfs, y_train)
y_pred_1 = logmodel.predict(X_test_sfs)
print(f1_score(y_test, y_pred_1))  # 0.6965

# The forward selection model doesn't really show a significant difference but it improves from 68% to 69%
