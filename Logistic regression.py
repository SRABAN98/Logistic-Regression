#Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Import the dataset
dataset = pd.read_csv(r"C:\Users\dell\OneDrive\Documents\Data Science\28th,29th\2.LOGISTIC REGRESSION CODE\Social_Network_Ads.csv")


#this dataset contians information of user and social network, those features are viz user-id,gender,age,salary,purchased
#social network has several business clients which can put their name into social networks and one of the client is car company, this company has newly lunched XUV in rediculous price or high price
#we will see which of the user in this social network are going to buy brand new XUV car
#Last column tells us user purchased the car yes-1 // no-0 & we are going to build the model that is going to predict if the user is going to buy XUV or not based on 2 variable based i.e age & estimated salary
#so our matrix of feature is only these 2 column & we gonna find some correlation b/w age and estimated salary of user and his decission to purchase the car [yes or no]
#so I need 2 index and I will remove rest of the index, for this I have to use slicing operator
#1 means - the user going to buy the car & 0 means - user is not going to buy the car


#splitting the dataset into I.V and D.V
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values


#splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)


#we are going to predict which users are going to buy the XUV, 
#Feature Scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test) 
#we mentioned feature scaling only to independent variable not to dependent variable at all


#data pre-processing done
#******************************************************************************************


#Next step is we are going to build the logistic regression model and apply this model into our dataset 
#This is linear model library that is why we called from sklear.linear_model
#Training the Logistic Regression model on the Training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)


#we have to fit the logistic regression model to our test set
#Predicting the Test set results
y_pred = classifier.predict(x_test)
#now we predict the y_pred table by passing x-test onto it, x-test we have age and salary , 
#if u look at the first observation this user is not be able to buy the XUV car but if you look at observation 7 then that user is going to buy the XUV car
#in this case logistic regression model classify that which users are going to buy the car or not 
#we build our logistic regression model and fit it to the training set & we predict our test set result 


#now we will use the confusion matrix to evalute
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# This is to get the Models Accuracy 
from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print(ac) 


# This is to get the Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
cr


bias = classifier.score(x_train, y_train)
bias

variance = classifier.score(x_test, y_test)
variance


#********************************************************************************************************************
#Now we gonna see the visulization 
# Visualising the Training set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(("red", "green")))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(("red", "green"))(i), label = j)
plt.title("Logistic Regression (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

#after execute this graph we will get the good graph & let's try to understand and analyse the graph step by step
#we have points like red point & green points, all these points are our observation of the training part
#each of the user's age is estimated at the x-axis & estimated salary are estimated at y-axis
#red points are the training set observation which the dependent variable purchased equal to 0
# green points are the training set observation which the dependent variable puchased equal to 1
#red points users are didn't able to buy the XUV & green points users are able to buy the XUV
#what we observed hear is users are young with low estimate salary actually didn't buy the XUV
#if you look at the users with older with high salary they will buy the XUV & XUV is family car so more older people are likely to buy this car
#now if you see some green points you can see in the red part , in this case even though older but due to less salary they are unable to buy car
#also some of the young people are also buy the car becuase they might be rich kid
#now what is the goal of classification & what classifer exactly do here & how this classifier will do for this business use case
#the main goal is classify the right users into right category that machine will do by logistic regression using s-curve
#the machine classify all dataset in 2 region, left region is to classify who not buy the car and the green is to classify who can buy the car
#logistic regression model ploted red pooint users are not going to buy the XUV and green point users are going to buy the XUV
#logistic regression will tell us that each user in the dataset is proper classified based on age & salary
#main important thing is these are the 2 prediction (red & green) separated by straight line & the straight line is called prediction boundry
#as we are talking logistic regression as linear model so we will required only for 2 variables & separated with straight line and logistic regressin is linear classifer
#we will see next, how non-linear classifier will separte won't be a straight line
#even though if you see the green point even though low salary they buy the XUV which is incorrect 
#if you see the green points are belongs to the red regions this happens becuae users are non-linear but we separate with linear model that's why we got this prediction 
#now we are looking visualiaton for training set & next we going to see the visualization for test set


#Visualising the Test set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(("red", "green")))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(("red", "green"))(i), label = j)
plt.title("Logistic Regression (Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

#let's see the graph, we get good graph we have plotted for the test data point 
#if you see the confustion matrix we saw 11 points are incorrectly predicted here, you can count that 
