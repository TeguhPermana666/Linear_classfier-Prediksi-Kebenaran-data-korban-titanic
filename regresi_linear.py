"""
algorithm in the topics:
-linear regresion
-classification
-clustring
-hidden markova blyat models
"""

#LINEAR REGRESION
"""
most basic dalam ml yang digunakan untuk memprediksi sebuah numerical values
->digunakan untuk memprediksi tingkat kelangsungan hidup penumpang dari titanic pada
data set
How to work?
Jika titik data terkait secara linier, kita dapat menghasilkan 
garis yang paling cocok untuk titik-titik ini dan menggunakannya
untuk memprediksi nilai masa depan.


"""
import matplotlib.pyplot as plt
import numpy as np
X=[1,2,2.5,3,4]
y=[1,4,7,9,15]
plt.plot(X,y,'ro')
plt.axis([0,6,0,20])
#refersher equation of a line in 2d
#y=mx + b
print(np.unique(X))
plt.plot(np.unique(X),np.poly1d(np.polyfit(X, y, deg=1))(np.unique(X)))
plt.show()
