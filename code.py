# Importing the libraries
from matplotlib import pyplot
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
import missingno as msno 

#Ignore Warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

# Importing the dataset
dataset = pd.read_csv('CE802_Ass_2019_Data.csv')
testset = pd.read_csv('CE802_Ass_2019_Test.csv')
dataset = dataset.fillna(dataset.F20.mean()) 
testset = testset.fillna(testset.F20.mean()) 
X = dataset.iloc[:, dataset.columns != 'Class' ].values
y = dataset.iloc[:, 20].values

#Analyzing the dataset
dataset.boxplot()

#Missing values F20
#msno.bar(dataset) 

print(dataset.Class.value_counts())
dataset.Class.value_counts().plot(kind = 'bar', title = 'class balance')

#Balancing
smote_balance = SMOTE(random_state = 42)                   
x_bal, y_bal = smote_balance.fit_sample(X, y) 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

#Kfolds
from sklearn.model_selection import RepeatedKFold
rkf = RepeatedKFold(n_splits=10, n_repeats=20, random_state=42)
# X is the feature set and y is the target
for train_index, test_index in rkf.split(X):
     print("Train:", train_index, "Validation:", test_index)
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state = 42)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy DT:",metrics.accuracy_score(y_test, y_pred))

#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print("Accuracy KNN:",metrics.accuracy_score(y_test, y_pred))

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 42)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print("Accuracy SVM:",metrics.accuracy_score(y_test, y_pred))

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 42)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print("Accuracy Kernel:",metrics.accuracy_score(y_test, y_pred))

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print("Accuracy Naive Bayes:",metrics.accuracy_score(y_test, y_pred))

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(random_state = 42)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print("Accuracy XGBoost:",metrics.accuracy_score(y_test, y_pred))

#Choose XGBoost as prediction model for training set
prediction = classifier.predict(testset.iloc[:, testset.columns != 'Class' ].values)
testset['Class'] = prediction 
predictionfile = pd.DataFrame(testset, columns=['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10',
 'F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','Class']).to_csv('prediction.csv')

# Test options and evaluation metric
num_folds = 10
scoring = 'accuracy'

models = []
models.append(('LR' , LogisticRegression()))
models.append(('LDA' , LinearDiscriminantAnalysis()))
models.append(('RF' , RandomForestClassifier()))
models.append(('KNN' , KNeighborsClassifier()))
models.append(('DT' , DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state = 42)))
models.append(('NB' , GaussianNB()))
models.append(('SVM' , SVC(kernel = 'linear')))
models.append(('Kernel' , SVC(kernel = 'rbf')))
models.append(('XGB' , XGBClassifier()))

# Evaluate each algorithm for accuracy
results = []
names = []
for name, model in models:
  kfold = KFold(n_splits=num_folds, random_state=42)
  cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)
  
  # Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#Predicted True or False
print(testset.Class.value_counts())
testset.Class.value_counts().plot(kind = 'bar', title = 'class balance')