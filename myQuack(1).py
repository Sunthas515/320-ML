
'''

2020 

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.


'''
import numpy as np
import pandas
import warnings

from sklearn import tree, neighbors, svm
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (10263047, 'Declan', 'Kemp'), (10482652, 'Callum', 'McNeilage') ]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    source = np.array(pandas.read_csv(dataset_path, header = None))
    X = source[:, 2:].astype(np.float32)
    y = np.array([1 if c == 'M' else 0 for c in source[:, 1]])
    
    return X, y

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DecisionTree_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    #Create Classifier
    base_clf = tree.DecisionTreeClassifier()
    params = {'min_samples_leaf' : list(range(1,10))}
    clf = GridSearchCV(base_clf, params, cv=5, iid=False)
    
     #Set up training data
    clf.fit(X_training, y_training)
    
    return clf.best_estimator_
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearrestNeighbours_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    # Create classifier
    base_clf = neighbors.KNeighborsClassifier()
    params = {'n_neighbors' : list(range(1,20,2))}
    clf = GridSearchCV(base_clf, params, cv=5, iid=False)
    # Set up the Training data
    clf.fit(X_training, y_training)
    
    return clf.best_estimator_

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    # Create Classifier
    base_clf = svm.SVC(gamma='scale')
    params = {'C' : [1000, 100, 10, 1, 0.1, 0.01, 0.001]}
    clf = GridSearchCV(base_clf, params, cv=5, iid=False)
    #Set up the Training data
    clf.fit(X_training, y_training)
    
    return clf.best_estimator_

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NeuralNetwork_classifier(X_training, y_training):
    '''  
    Build a Neural Network classifier (with two dense hidden layers)  
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    #Create Classifier
    base_clf = MLPClassifier(max_iter=1000)
    params = {'hidden_layer_sizes' : [i*np.ones(3).astype(int) for i in range(5,30,5)]}
    clf = GridSearchCV(base_clf, params, cv=5, iid=False)
    #Set up the Training data
    with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                    module="sklearn")
            clf.fit(X_training, y_training)
    
    return clf.best_estimator_

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ## AND OTHER FUNCTIONS TO COMPLETE THE EXPERIMENTS
    ##         "INSERT YOUR CODE HERE"    
    #raise NotImplementedError()
def NN_PreProcessing(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def DecisionTree_Experiment(X_train, X_test, y_train, y_test):
    clf = build_DecisionTree_classifier(X_train, y_train)
    acc = clf.score(X_test, y_test)
    l = clf.get_params()['min_samples_leaf']
    print('Decision Tree:          {:2.1%} accurate with {:d} minimum samples per leaf'.format(acc,l))
    
def NearestNeighbours_Experiment(X_train, X_test, y_train, y_test):
    clf = build_NearrestNeighbours_classifier(X_train, y_train)
    acc = clf.score(X_test, y_test)
    n = clf.get_params()['n_neighbors']
    print('Nearest Neighbours:     {:2.1%} accurate with {:d} nearest neighbours considered'.format(acc,n))    
    
def SupportVectorMachine_Experiment(X_train, X_test, y_train, y_test):
    clf = build_SupportVectorMachine_classifier(X_train, y_train)
    acc = clf.score(X_test, y_test)
    C = clf.get_params()['C']
    print('Support Vector Machine: {:2.1%} accurate with {:d} as C value'.format(acc,C))

def NeuralNetwork_Experiment(X_train, X_test, y_train, y_test):
    X_trainS, X_testS = NN_PreProcessing(X_train, X_test)
    clf = build_NeuralNetwork_classifier(X_trainS, y_train)
    acc = clf.score(X_testS, y_test)
    n = sum(clf.get_params()['hidden_layer_sizes'])
    print('Neural Network:         {:2.1%} accurate with {:d} hidden neurons'.format(acc, n))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    # Write a main part that calls the different 
    # functions to perform the required tasks and repeat your experiments.
    # Call your functions here
    ##         "INSERT YOUR CODE HERE"      
    #Print names
    print(my_team())
    
    # Data pre-processing
    X, y = prepare_dataset('./medical_records(1).data')
    
    #Create test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    DecisionTree_Experiment(X_train, X_test, y_train, y_test)
    NearestNeighbours_Experiment(X_train, X_test, y_train, y_test)
    SupportVectorMachine_Experiment(X_train, X_test, y_train, y_test)
    NeuralNetwork_Experiment(X_train, X_test, y_train, y_test)
    