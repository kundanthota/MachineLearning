import pandas as pd
f=[int(i) for i in range(1,95)]
data=pd.read_csv('INS_test.csv',names=f)
data=data[1:]
data.index = data.index.map(str)
data=data.astype(str)
ls=[]
for i in range(1,len(data)):
    st="ex"+str(data[1][i])+"|f  "
    for j in range(2,len(data.columns)):
        if data[j][str(i)]!='0':
            st=st+str(j)+":"+str(data[j][str(i)])+" "
    ls.append(st)
with open('Test_ISN.txt', 'w+') as f:
    for item in ls:
        f.write("%s\n" % item)
    f.close()