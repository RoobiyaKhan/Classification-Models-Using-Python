#logistic regression
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Feature Scaling  #for accurate predic
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
'''sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)''' #since already its categorical dep variable

#fitting logistic regression to the training set  #logistic is linear classifier
                     #(since here in 2D,our 2 categories of users will be seperated by a straight line)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0) #classifier is the object of logistic reg class
classifier.fit(x_train, y_train)

#predicting the test set results
y_pred = classifier.predict(x_test)    #vector of predictions-of each of test set observation

# making the confusion matrix
from sklearn.metrics import confusion_matrix #its not a class,its a function
                                           #(class is in capital letter at the beginning of words)
cm = confusion_matrix(y_test, y_pred)                                           
cm

#visualizing the training set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train #giving local variable names
x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop =x_set[:, 0].max() + 1, step = 0.01),#for age
                    np.arange(start = x_set[:, 1].min() - 1, stop =x_set[:, 1].max() + 1, step = 0.01))#for salary
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green')))# ravel() flattens the array into 1D and T is transposition to make it a vector
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('logistic regression(training set)')
plt.xlabel('age')
plt.ylabel('estimated salary')
plt.legend()
plt.show()

#visualizing the test set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test #giving local variable names
x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop =x_set[:, 0].max() + 1, step = 0.01),#for age
                    np.arange(start = x_set[:, 1].min() - 1, stop =x_set[:, 1].max() + 1, step = 0.01))#for salary
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green')))# ravel() flattens the array into 1D and T is transposition to make it a vector
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j) #c is a parameter for colour;The method ListedColormap() provides computer codes
                       #for coloring. First 3 entries are for  RGB and the last entry stands for transparency.
                                                    #eg:from matplotlib.colors import ListedColormap
                                                     #i=0
                                                     #ListedColormap(('red', 'green'))(i)
                                              #output:(1.0, 0.0, 0.0, 1.0)
                                                     #i=1                                                  
                                                     #ListedColormap(('red', 'green'))(i)
                                              #output:(0.0, 0.50196078431372548, 0.0, 1.0)"""
plt.title('logistic regression(test set)')
plt.xlabel('age')
plt.ylabel('estimated salary')
plt.legend()
plt.show()

#We can extract probability values and create a prediction based on a specified cut of value.
prob_pred= classifier.predict_proba(x_test)
y_pred = 1*(prob_pred[:,0] > 0.6) # giving cut_of_value = 0.6 
plt.clf() # to remove previous plot
plt.hist(prob_pred[:, 0])
plt.show()