import pandas as pd
import numpy as np
from operator import sub,mul
data=pd.read_csv('DWH Training.csv',names=['height','weight','gender'])
data.gender=data['gender'].replace([1,-1],[-1,1]) #to change the gender classes 
test=pd.read_csv('DWH_test.csv',names=['height','weight','gender','distance'])
test.gender=test['gender'].replace([1,-1],[-1,1])#to change the gender classes
test=test[['height','weight','gender']]
#function initial_weights_bias() is to initialize weights and bias using Nearest centroid classifier
def initial_weights_bias(data):
    negative_w=[data[data.gender==-1].height.mean(),data[data.gender==-1].weight.mean()]
    positive_w=[data[data.gender==1].height.mean(),data[data.gender==1].weight.mean()]
    w_init=[2*i for i in list(map(sub,positive_w,negative_w))]
    b_init=sum(np.square(negative_w))-sum(np.square(positive_w))
    w_init.append(b_init)
    return w_init
#function sum_of_weights is to calculate the c*summation(-x_i*y_i),c*summation(-y_i)
def sum_of_weights(c,rand,grad):
    init1,init2,init3=[],[],[]
    for h in rand.itertuples():
        if h.gender*(h.height*grad[0]+h.weight*grad[1]+grad[2])<1:
            init1.append(-h.gender*h.height)
            init2.append(-h.gender*h.weight)
            init3.append(-h.gender)
    return c*sum(init1),c*sum(init2),c*sum(init3)
#function grad_vector() is to minimize the error
def grad_vector(data,w_b,B,c):
    grad=w_b
    for i in range(1,10001):
        rand=data.sample(n=B)
        summed_weights=sum_of_weights(c,rand,grad)
        lam=1/i
        grad[0]-=lam*(grad[0]+summed_weights[0])
        grad[1]-=lam*(grad[1]+summed_weights[1])
        grad[2]-=lam*(summed_weights[2]) 
    return grad
#function accuracy() is to calculate accuracy of test data
def accuracy(grad,test_data):
    acc=0
    for h in test_data.itertuples():
        if (h.height*grad[0]+h.weight*grad[1]+grad[2]<0 and h.gender==-1) or (h.height*grad[0]+h.weight*grad[1]+grad[2]>0 and h.gender==1):
            acc+=1
    return (acc*100/len(test_data))
if __name__=="__main__":
    w_b=initial_weights_bias(data)
    grad=grad_vector(data,w_b,50,1)
    print(grad)
    print(accuracy(grad,test))    