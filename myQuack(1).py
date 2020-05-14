
'''

2020 

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.


'''


<<<<<<< HEAD

=======
import numpy as np
<<<<<<< HEAD
>>>>>>> callum-code
=======
from sklearn import tree, neighbors, svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
>>>>>>> callum-code
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
<<<<<<< HEAD
<<<<<<< HEAD
#    return [ (1234567, 'Ada', 'Lovelace'), (1234568, 'Grace', 'Hopper'), (1234569, 'Eva', 'Tardos') ]
=======
#    return [ (10263047, 'Declan', 'Kemp'), (10482652, 'Callum', 'McNeilage') ]
>>>>>>> callum-code
    raise NotImplementedError()

=======
    return [ (10263047, 'Declan', 'Kemp'), (10482652, 'Callum', 'McNeilage') ]
>>>>>>> callum-code
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
<<<<<<< HEAD
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()
=======
    '''    
    # Create one-dimensional numpy array using class label of X[i,:]
    X = np.genfromtxt(dataset_path, dtype ='U' ,delimiter=",")
    y = X[:,1]
    
    for i in range(len(y)):
        if y[i] == 'M':
            y[i] = 1
        else:
            y[i] = 0
            
    y = np.asarray(y, dtype='i')
    
    # Create two-deimensional numpy array X
    X = np.genfromtxt(dataset_path, dtype = 'f', delimiter=',')
    
    X = np.delete(X, 1, 1)
    
    return(X,y)

>>>>>>> callum-code

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
    ##         "INSERT YOUR CODE HERE"    
    
    #Create Classifier
    classifier = tree.DecisionTreeClassifier()
    
    #Enter Parameters
    params = [
            {
                    'splitter': ['best', 'random'],
                    'max_depth': np.linspace(1, 100, 100)
                    }
            ]
    
    #Use gridSearch to estimate best value
    clf = GridSearchCV(classifier, params)
    
    #Set up training data
    clf.fit(X_training, y_training)
    
    return clf
    
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
    ##         "INSERT YOUR CODE HERE"    
    
    # Create Classifier
    classifier = neighbors.KNeighborsClassifier()
    # Enter Parameters
    params = [
            {
                    'n_neighbours': np.arange(20) + 1,
                    'leaf_size': np.arange(50) + 1
                    }
            ]
    
    # Use gridSearch to estimate best value
    clf = GridSearchCV(classifier, params)
    
    # Set up the Training data
    clf.fit(X_training, y_training)
    
    return clf

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
    ##         "INSERT YOUR CODE HERE"    
    
    # Create Classifier
    classifier = svm.SVC()
    
    # Enter parameters
    params = [
            {
                    'C': np.logspace(-3, 3, 7),
                    'kernel': ['linear']
                    },
            {
                    'C': np.logspace(-3, 3, 7),
                    'gamma': np.logspace(-4, 4, 9),
                    'kernel': ['rbf']
                    }
            ]
    
    # Use gridSearch to estimate best value
    clf = GridSearchCV(classifier, params)
    
    #Set up the Training data
    clf.fit(X_training, y_training)
    
    return clf

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
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ## AND OTHER FUNCTIONS TO COMPLETE THE EXPERIMENTS
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
<<<<<<< HEAD
    pass
=======
>>>>>>> callum-code
    # Write a main part that calls the different 
    # functions to perform the required tasks and repeat your experiments.
    # Call your functions here

<<<<<<< HEAD
    ##         "INSERT YOUR CODE HERE"    
<<<<<<< HEAD
    raise NotImplementedError()
=======
    print(prepare_dataset('./medical_records(1).data'))
>>>>>>> callum-code
=======
    ##         "INSERT YOUR CODE HERE"  
>>>>>>> callum-code
    
    #Print names
    print(my_team())
    
    # Data pre-processing
    X, y = prepare_dataset('./medical_records(1).data')
    
    #Create test data
    X_trainer, X_tester, y_trainer, y_tester = train_test_split(X, y, test_size=0.3)
    
    #Create classifiers
    classifiers = [[build_DecisionTree_classifier, "Decision Tree"],
                   [build_NearrestNeighbours_classifier, "Nearest Neighbours"],
                   [build_SupportVectorMachine_classifier, "Support Vector Machine"]]
    
    #Output each classifier's values
    for function, name in classifiers:
        classifier = function(X_trainer, y_trainer)
        #Print Outputs
        print(name, "Best Parameters:", classifier.best_params_)
        #Generate report for training data
        predict_training = classifier.predict(X_trainer)
        print(name, "Classification Report:")
        print(classification_report(y_trainer, predict_training))
        # Generate report for test data
        predict = classifier.predict(X_tester)
        print(name, "Test Data Classification Report:")
        print(classification_report(y_tester, predict))