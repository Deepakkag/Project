import pandas as pd
import numpy as np
import os
print("Librarie imported succesfully!")


# reading the data
df = pd.read_csv("DS_iris.csv")
df.head()


# check the data type in the data set
df.info()


# check for the null values
df.isnull().sum()


# we don't have any null value in the data set
# converting the all the data type to numeric, so we are changing the data type of species  column

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# changing the data type using the label encoder

df['species'] = le.fit_transform(df['species'])
df.head()


from sklearn.model_selection import train_test_split

# For training considering the 70% of data
# For testing considering the 30% of the data

X = df.drop(columns= ['species'])
Y = df['species']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30)


# we are using the logistic regression to train the model and test the data set

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# train the data set using the model

model.fit(x_train, y_train)


# checking the accuracy of the above model used

print("Accuracy", model.score(x_test, y_test) * 100)

# we are getting an accuracy of 97.8% with the classification model named as Logistic Regression

