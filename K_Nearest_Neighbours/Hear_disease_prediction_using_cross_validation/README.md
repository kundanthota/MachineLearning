## KNN using t-times k-fold cross validation
Data set is divided into k folds in which everytime one fold will be the test data and remaining folds are training data.Cross-validation, sometimes called rotation estimation, is a model validation technique for assessing how the results of a statistical analysis will generalize to an independent data set. It is mainly used in settings where the goal is prediction, and one wants to estimate how accurately a predictive model will perform in practice.

#### prerequisites  
1.Pandas
2.Numpy

#### Algorithm
1. Find the euclidean distances of features of test point with features all data points of train data.
2. sort the data with respect to distance in ascending order.
3. classify the test data according to votes by K train data points.

In the problem I took k= sqrt(len(train_data))

