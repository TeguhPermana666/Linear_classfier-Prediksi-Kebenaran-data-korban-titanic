import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#exploring,cleaning,selecting data yang sesuai

#load data
dftrain=pd.read_csv('core_learn/train.csv')#xtrain
dftesting=pd.read_csv('core_learn\eval.csv')#xeval
print(dftrain.head())
print(dftesting.head())
#masukan data y
y_train=dftrain.pop('survived')
y_eval=dftesting.pop('survived')
print(y_train)
print(y_eval)
print(dftrain)
print(dftrain.describe())
print(dftrain.shape)
print(dftrain.dtypes)
# dftrain.age.hist(bins=20)
# plt.show()

# dftrain.sex.value_counts().plot(kind='barh')
# plt.show()

pd.concat([dftrain,y_train],axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('survived')
plt.show()

#dari analisis diagram tersebut kita dapat simpulkan
"""
- Sebagian besar penumpang berusia 20-an atau 30-an
- Sebagian besar penumpang adalah laki-laki
- Sebagian besar penumpang berada di kelas "Ketiga"
- Wanita memiliki peluang bertahan hidup yang jauh lebih tinggi
"""
