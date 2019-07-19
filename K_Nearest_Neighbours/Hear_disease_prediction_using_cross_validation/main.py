import pandas as pd
import numpy as np
from operator import sub
data=pd.read_csv('heart.csv')
data['target']=data['target'].replace(0,-1)
data=data[['age','trestbps','chol','thalach','target']]
#function calculate_distance is to calculate the Euclidean distance between features
def calculate_distance(train_data,test_data):
    distance_list=list(map(sub,train_data,test_data))
    distance_list=distance_list[:len(distance_list)-1]
    return np.sqrt(sum([i**2 for i in distance_list]))   
#function prediction the class using k nearest neighbours
def prediction_knn(test_data,k):
    calculated_distance_list=[]
    predictor=0
    for train_data in train.itertuples():
        calculated_distance_list.append((calculate_distance(train_data,test_data),train_data.target))
    calculated_distance_list.sort(key=lambda var:var[0])
    for i in range(k):
        predictor+=calculated_distance_list[i][1]
    return(1 if predictor>1 else -1)
#function accuracy is to find the accuracy of test data with respect to train data
def accuracy(k):
    count=0
    for test_data in test.itertuples():
        if(test_data.target==prediction_knn(test_data,k)):
            count+=1
    return 100*(count/len(test))
if __name__== "__main__":
    n=int(np.sqrt(len(data)))
    acc=0
    #t times t fold cross validation
    for i in range(n):
        test=data[:n]
        train=data[n:]
        data=pd.concat([train,test],axis=0)
        acc+=accuracy(n)
    print(f"Accuracy of {n} Nearest Neighbours using {n} times {n} fold cross validation is : {acc/n}" )
