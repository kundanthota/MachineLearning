## Liblinear
LIBLINEAR is an open source library for large-scale linear classification. It supports logistic regression and linear support vector machines.
#### Prerequisites
1.Liblinear Package , ref :https://github.com/cjlin1/liblinear for installation and running.
2.numpy
3.pandas
#### Algorithm
1. consruct data in libsvm format or python format or scipy format
2. train the data with some c-type parameters
3. After training the problem use model to predict the test data 
#### parameters and functions 
1.s0,s1,s2... are types of solvers
2.-c: cost parameter ex: -c 4 cost of 4
3. -B: Bias ex: -B 10 bias of 10
4.-w: weights
5.-v: cross validation ex: -v 5 cross validation with 5 folds
6.problem function is to load problem in respective formats.
7.parameter function is to load c-type parameters.
8.train function is to train the problem.
9.predict function is for the prediction of test data with trained model.
10.get_decfun returns the weights and bias
