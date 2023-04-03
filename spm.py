#logistic regression is good for binary data

# import lib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer	#we have to covert text data to numerical values
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load data
data = pd.read_csv('mail_data.csv')
print(data)

# finding null values
print(data.isnull().sum())

print(data.shape)

# label spam mail as 0, spam mail is 1
d1 = pd.get_dummies(data.Category)
print(d1)
 
nd1 = pd.concat([data, d1], axis="columns")
print(nd1)

X = data['Message']
Y = data['Category']

print(X)
print(Y)

#training and testing of data
x_train, x_test, y_train , y_test = train_test_split( X, Y, test_size=0.2, random_state=2)

print(X.shape)
print(x_train.shape)
print(x_test.shape)

#fearure extraction
fe1= TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')

xtrainf = fe1.fit_transform(x_train)
xtestf = fe1.transform(x_test)

print(xtrainf)
print(xtestf)

# model and fit
model = LogisticRegression()
model.fit(xtrainf, y_train)

#prediction on training data
prediction_on_training_data = model.predict(xtrainf)
acc_otd= accuracy_score(y_train, prediction_on_training_data)
print(acc_otd)

# prediction on test data
prediction_on_test_data = model.predict(xtestf)
acc_otd1 = accuracy_score(y_test, prediction_on_test_data)
print(acc_otd1)

#building predictive system

input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]
inpm1 = ["07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile + free camcorder. Please call now 08000930705 for delivery tomorrow"]

#convert test to features vectors
f1 = fe1.transform(input_mail)
f2 = fe1.transform(inpm1)

#making prediction
pred = model.predict(f1)
pred1 = model.predict(f2)
print(pred)
print(pred1)
