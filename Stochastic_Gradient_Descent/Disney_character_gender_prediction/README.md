## Stochastic Gradient Descent
#### Algorithm:
1: initialize (b, w)_0 (through Nearest Centroid Classifier)
2: for t = 1 : T do
3: Randomly select B many data points
4: Denote their indexes by Ic{1, . . . , n} (i.e.,|I| = B)
5: Update (b, w)_t+1 :=
(b, w)_t - ?_t gradient(1/2 |w|^2 + c summation(max (0,1-y_i(W^Tx_i+b))) where i in I
6: end for

## Nearest Centroid Classifier
There are two classes in the current dataset with two input features.  
#### Algorithm:
1. Find centroids of two classes
2. find w= 2*(c_1 - c_2) where c_1,c_2 are two classes
3. find b= ||c_2||^2-||c_1||^2
