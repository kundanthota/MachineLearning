import pandas as pd
from operator import add,truediv,mul,sub
import numpy as np
data=pd.read_csv("Admission_Predict.csv")
data.columns=['no','gre','toefl','rating','sop','lor','gpa','research','admit']
data=data[['gre','toefl','rating','sop','lor','gpa','research','admit']]
#calculating means of all columns
cal_means=list(data.mean())
#finding scale factors and bias
def sf_bias(cal_means,data):
    numerator,denominator=[0 for i in range(6)],[0 for i in range(6)]
    for value in data.itertuples():
        numerator=list(map(add,numerator,[(value.gre-cal_means[0])*(value.admit-cal_means[7]),(value.toefl-cal_means[1])*(value.admit-cal_means[7]),(value.rating-cal_means[2])*(value.admit-cal_means[7]),(value.sop-cal_means[3])*(value.admit-cal_means[7]),(value.lor-cal_means[4])*(value.admit-cal_means[7]),(value.gpa-cal_means[5])*(value.admit-cal_means[7]),(value.research-cal_means[6])*(value.admit-cal_means[7])]))
        denominator=list(map(add,denominator,[(value.gre-cal_means[0])**2,(value.toefl-cal_means[1])**2,(value.rating-cal_means[2])**2,(value.sop-cal_means[3])**2,(value.lor-cal_means[4])**2,(value.gpa-cal_means[5])**2,(value.research-cal_means[6])**2]))
    sf=list(map(truediv,numerator,denominator))
    bias=cal_means[7]-sum(map(mul,cal_means,sf))
    return sf,bias
#find r square error 
def r_squared(sf,bias,data):
    actual=data['admit'].tolist()
    actual_minus_actualmean=[(i-np.mean(actual))**2 for i in actual]
    data=data[['gre','toefl','rating','sop','lor','gpa','research']]
    input_features=data.values.tolist()
    predicted_minus_predictedmean=[]
    for x,y in zip(input_features,actual):
        predicted=sum(list(map(mul,x,sf)))+bias
        predicted_minus_predictedmean.append((predicted-np.mean(actual))**2)    
    r_2= 1-sum(actual_minus_actualmean)/sum(predicted_minus_predictedmean)
    return r_2
if __name__ == "__main__":
    sf,bias=sf_bias(cal_means,data)
    print(sf)
    print(bias)
    print(r_squared(sf,bias,data))