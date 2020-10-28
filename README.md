# machine-learning-getting-started
import sys
import scipy
import numpy 
import pandas
import matplotlib
import sklearn
import pandas
from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.model_selection importtrain_test_split
from sklearn_.model_selection import cross_val_score
from sklear.model_selection import StratifiedKFold
from sklearn.metrices import classification_report
from sklearn.metrices import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVM
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection

# importig data set
url = "https://raw.githubusercontent.com/jbrownlee/Dataseta/master.iris.csv"
names = ['sepal-length' , 'sepal-width' , 'petal-length' , 'class']
datasets= read_csv(url, names = names)

#dimensions of the dataset
print(dataset.shape)

# take a peak at data
print(dataset.head(20))

#statistical summary
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())

#visuvalising the data
# univariant plots - box and whisper plots
dataset.plot(kind='box' , subplot = True , layout = (2,2) , sharex = False , sharey = False)

#histogram of the variable
dataset.hist()
pyplot.show()
 
 # multivariate plots
 scatter_matrix(dataset)
 pyplot.show()
 
 # creating a validation set
 # splitting the dataset
 array = dataset.values
 X = array[: , 0:4]
 Y = array[: , 4]
 x_train , X_validation , y_train , Y_validation = train_split(X,Y , test_size = 0,2 , random_state = 1)
 
# logistic regression
# linear discriminant analysis
# k - N neighbors 
# classification and regression trees
# gaussian naive bayes
# support Vector machines

#building models
model = []
models.append()'LR' , LogisticRegression(solver='liblinear' , multi_class='ovr')))
models.append(('LDA,LinearDIscriminantAnalysis()))
models.append(('KDN' , KNneighborClassifier()))
models.append()'NB' , GaussianNB()))
models.append(('SVM' , SVC (gamma= 'auto')))

#evaluate the created models
results = []
names = []
for name , model in models
    kfold = StratifiedKFold(n_splits=10, random_state=1)
    cv_results = cross_val_score(model , X_train ,Y_train , cv=fold , scoring ='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (names , cv_results.mean(), cv_results.std()))
    
# compare our models 
pyplot.boxplot(results , labels = names)
pyplot.title('algebra compariosion')
pyplot.show()

# make a prediction
model = SVC(gamma= 'auto')
model.first(X_train , Y_train)
predictions = model.predict(X_validation)

# evaluating our predictions
print(accuracy_score(Y_validation , predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation , predictions))
