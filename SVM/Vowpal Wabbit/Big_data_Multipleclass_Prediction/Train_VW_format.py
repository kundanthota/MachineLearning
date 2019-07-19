import pandas as pd
f=[int(i) for i in range(1,95)]
data=pd.read_csv('INS_training.csv',names=f)
data.index = data.index.map(str)
data=data.astype(str)
data=data[1:]
data=data.replace(["Class_"+str(i) for i in range(1,10)],[i for i in range(1,10)])
ls=[]
for i in range(1,len(data)):
    st=str(data[94][str(i)])+" "+"ex"+str(i)+"|f "
    for j in range(1,len(data.columns)):
        if data[j][str(i)]!='0':
            st=st+str(j)+":"+str(data[j][str(i)])+" "
    ls.append(st)
with open('Train_ISN.txt', 'w+') as f:
    for item in ls:
        f.write("%s\n" % item)
    f.close()