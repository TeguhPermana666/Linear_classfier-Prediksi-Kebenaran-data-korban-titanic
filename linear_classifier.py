#supervised model-Lestimator regresion
"""
train=>feed the model,yg digunakan untuk mendevelope dan learn 
ukurannya selalu lebih besar daripda testing

testing=>digunakan untuk meevaluate sebuah model dan bagimana ia
perform,set dari data evaluate tidak di trained jika sebuah set 
tersebut untuk melakukan evaluate dikarenakan agar data tersebut 
sperti data baru yang harus di kenali modle yang dilatih pada train data

point dari suatu model adalah mengablekan pembuatan prediction pada new data
yang mana data yang tidak pernah diliat sebelumnya.

jika kita melakukan sebuah pengujian tidak yakin untuk menghasilkan hasil acuracy yang begitu 
akurat, tidak terlalu simple untuk model menyimpan atau merekam sebuah training data, maka dari itu
train and testing harus dipisah. 
"""
"""
columns terdapat dua informasi yang berbeda 
1.categorical
=>anything its no numeric 
2.numeric
=>numbers
"""
#convert categorical data into numerical
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#exploring,cleaning,selecting data yang sesuai

#load data
dftrain=pd.read_csv('core_learn/train.csv')#xtrain
dftesting=pd.read_csv('core_learn\eval.csv')#xeval
# print(dftrain.head())
# print(dftesting.head())

#masukan data y
y_train=dftrain.pop('survived')
y_eval=dftesting.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns=[]

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()# gets a list of all unique values from given feature column
    feature_columns.append(tf.feature_column.sequence_categorical_column_with_vocabulary_list(feature_name, vocabulary))
print(vocabulary.data) 

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
print(feature_columns)
print(dftrain.dtypes)
#append ->membuat object kedalam model yang bisa memetakan string  kedalam bilangan 

"""
untuk train data
model data =>dibagi dalam 32 batches
jadi tidak satu data frame dimasukan kedalam model

batch akan dimasukan kedalam model beberapa kali seiring banyaknya epoch
=>input function
tf.data.dataset ibject=>input function bisa untuk merubah current pandas dataframe 
kedalam object
"""
# print(dftrain['sex'].dtypes)#ini hasilnya dia dalam bentuk object tipenya
# dataset=tf.data.Dataset.from_tensor_slices((dict(dftrain),y_train))#dia pada sex menghasilkan tfstring,yg berarti semua data sudah menjadi object. 
# dataset=dataset.shuffle(1000)
# print(dataset,"\n\nBATCH:\n\n\n\n")
# dataset=dataset.batch(32).repeat(10)
# print(dataset)
# print(dict(dftrain))#nilai-nilainya dipisah berdasarkan jenis di setiap columnnya
def make_input_fn(data_df,label_df,num_epoch=10,shuffle=True,batch_size=32):
    def input_function():#inner function akan return ke make input
        # create tf.data.Dataset object with data and its label
        ds=tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds=ds.shuffle(1000)#randomize order of data
        ds=ds.batch(batch_size).repeat(num_epoch)# split dataset into batches of 32 and repeat process for number of epochs
        return ds
    return input_function
# here we will call the input_function that was returned to us to get a dataset object we can feed to the model
train_input_fn = make_input_fn(dftrain, y_train)  
eval_input_fn = make_input_fn(dftesting, y_eval, num_epoch=1, shuffle=False)
print(train_input_fn)
print(eval_input_fn)
#creating the model => linear estimator dengan memanfaatkan linear regresion algorithm
linear_est=tf.estimator.LinearClassifier(feature_columns=feature_columns)
#training model
linear_est.train(train_input_fn)#train
result=linear_est.evaluate(eval_input_fn)
print(result)
print(result['accuracy']*100)#keganti trs tergantung dari overfitting dan underfitting dalam pengenalan modelnya

#predictions=>membuat prediksi terhadap nilai y->kelangsungan hidup manusia kafir

predcts_dicts=list(linear_est.predict(eval_input_fn))
probs=pd.Series([pred['probabilities'][1] for pred in predcts_dicts])
print(probs)
probs.plot(kind='hist',bins=20,title='Predicted probabilites')
plt.show()