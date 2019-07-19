import pandas as pd 
train=pd.read_csv('INS_training.csv')
test=pd.read_csv('pred_squared.txt',sep=" ",names=['label','id'])
train=train.replace(["Class_"+str(i) for i in range(1,10)],[i for i in range(1,10)])
count=0
for i,j in zip(train['target'],test['label']):
    if i==j:
        count+=1
print(f"Accuracy is {count*100/len(test)}")