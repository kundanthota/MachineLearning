## Multiple regression
Equation : Y = a + b1*X1 + b2*X2 + ... + bp*Xp where x1,x2......,xp are independent variables and Y is dependent variable. b1,b2,...,bp are scaling factors and a is bias
 R^2 and Residual variance :R-Square, also known as the Coefficient of determination is a commonly used statistic to evaluate model fit. R-square is 1 minus the ratio of residual variability. When the variability of the residual values around the regression line relative to the overall variability is small, the predictions from the regression equation are good. For example, if there is no relationship between the X and Y variables, then the ratio of the residual variability of the Y variable to the original variance is equal to 1.0. Then R-square would be 0. If X and Y are perfectly related then there is no residual variance and the ratio of variance would be 0.0, making R-square = 1.

#### prerequisites 
1.pandas
2.numpy

#### Algorithm
1. Find the means of all independent features. 
2. find variance,co-variance, standard deviation of the data.
3. find the r^2 score of the predicted dependent feature withrespect to actual dependent feature.

