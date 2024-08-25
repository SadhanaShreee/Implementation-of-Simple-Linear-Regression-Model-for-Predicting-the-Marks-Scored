# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

### AIM:
## To write a program to predict the marks scored by a student using the simple linear regression model.

### Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

### Algorithm
## 1. Import the standard Libraries.
## 2. Set variables for assigning dataset values.
## 3. Import linear regression from sklearn.
## 4. Assign the points for representing in the graph.
## 5. Predict the regression for marks by using the representation of the graph.
## 6. Compare the graphs and hence we obtained the linear regression for the given datas.








## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SADHANA SHREE B
RegisterNumber: 212223230177

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
df = pd.read_csv('student_scores.csv')
print(df.head())
print(df.tail())
df.info()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
X_train.shape
X_test.shape
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
#Graph plot for training data
import matplotlib.pyplot as plt
plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,reg.predict(X_test),color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
mse = mean_squared_error(X_test, Y_pred)
print('MSE = ', mse)
mae = mean_absolute_error(X_test, Y_pred)
print('MAE = ', mae)
rmse = np.sqrt(mse)
print("RMSE= ", rmse)
```

### Output:
### df.head()
![df head](https://github.com/user-attachments/assets/c8d77a50-3b46-43e0-9609-0ed257d5269b)

### df.tail()
![df tail](https://github.com/user-attachments/assets/3539ba43-91ef-4656-88bd-8ab012346d85)

### df.info()
![3](https://github.com/user-attachments/assets/1e96f89e-2879-45df-993d-92efa33114b9)


### X and Y Values
![4](https://github.com/user-attachments/assets/12a713a4-593c-4521-8d6a-fd347e338910)

### Prediction Values of X and Y
![7](https://github.com/user-attachments/assets/cc1be360-ed2a-458f-8a5e-3259442fa7d4)

### Training Set
![8](https://github.com/user-attachments/assets/ee287241-6b9e-4aa8-9459-232a83940f7a)

### Testing Set
![9](https://github.com/user-attachments/assets/86818dc0-81b5-4493-a270-b0117dd12bef)

### MSE,MAE and RMSE
![10](https://github.com/user-attachments/assets/5648f501-0c21-44de-a4b8-70aacb9c9ccc)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
