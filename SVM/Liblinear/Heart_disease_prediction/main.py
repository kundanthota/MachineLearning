from liblinearutil import *
import pandas as pd
import numpy as np
data=pd.read_csv('heart.csv')
cols=data.columns.values.tolist()
#to convert feature column indices to numbers
data.columns=[i for i in range(1,len(cols)+1)]
if __name__ == "__main__":
    n=int(len(data)/17)
    accuracy=0
    #t times t fold cross validation
    for i in range(n):
        test_data=data[:n]
        train_data=data[n:]
        data=pd.concat([train_data,test_data],axis=0)
        y_train=[]
        x_train=[]
        x_test=[]
        y_test=[]
        for values in train_data.itertuples():
            #to load labels of train_data as a list
            y_train.append(values[len(cols)])
            #to load features of train_data as a list of dictionaries
            x_train.append({i:values[i] for i in range(1,len(cols))})
        for values in test_data.itertuples():
            #to load labels of test_data as a list
            y_test.append(values[len(cols)])
            #to load features of test_data as a list of dictionaries
            x_test.append({i:values[i] for i in range(1,len(cols))})
        #to load labels and features
        prob=problem(y_train,x_train)
        #parameter function takes parameters that can be used to train our data
        param=parameter('-s 2 -c 2')
        #train function is to train the model with values returend by problem and parameter 
        model=train(prob,param)
        #predict function returns predicted values, accuracy, predicted lables of test data using our trained model
        p_val,p_acc,p_lab=predict(y_test,x_test,model)  
        accuracy+=p_acc[0]
    print(f"Accuracy through {n} Times {n} Fold cross validation is : {accuracy/n}")