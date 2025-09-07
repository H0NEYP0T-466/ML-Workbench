import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


data={  
    'name':["Ali","Veli","Ayse","Fatma","Ahmet","Mehmet","Can","Cem","Deniz","Ece"],
    'age':[23,45,12,34,22,33,44,55,66,77],
    'spendings':[200,300,400,500,600,700,800,900,1000,1100]
}
df=pd.DataFrame(data)

model=KMeans(n_clusters=3,random_state=42,n_init=10)
X=df[['age','spendings']]
model.fit(X)
df['cluster']=model.fit_predict(X)
print(df)
plt.scatter(df['age'],df['spendings'],c=df['cluster'])
plt.xlabel('Age')
plt.ylabel('Spendings')
plt.show()

