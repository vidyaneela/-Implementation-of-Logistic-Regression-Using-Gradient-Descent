# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages required.
2.Read the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary and predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Vidya Neela.M
RegisterNumber:  212221230120
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:
![196483263-1df9013b-f249-4c69-94cc-7afd26a0028d](https://user-images.githubusercontent.com/94169318/200604619-bbc626b8-88a1-4e94-9eab-5e8bdcd45b6c.png)
![196483351-77eeb439-5d5f-44ed-beee-572660a918a0](https://user-images.githubusercontent.com/94169318/200604697-2fa7a6ae-fffb-4593-b56a-895d69d6d1eb.png)
![196483389-878c03ce-5d44-4442-aa8a-883dc87e2da7](https://user-images.githubusercontent.com/94169318/200604761-a5975c81-408d-4fe6-bf6a-17d4840108d6.png)
![196483418-6ee83be3-b8aa-4156-a8a1-a5b4ded626bd](https://user-images.githubusercontent.com/94169318/200604804-fb275bda-0e34-438d-8528-7de09f6a7e0a.png)
![196483446-06824a7a-4d26-49d4-aad9-c34ff731db21](https://user-images.githubusercontent.com/94169318/200604891-954ec55a-8230-408d-9b36-84070e380916.png)
![196483474-cc236bfc-14d0-4e95-99e8-690aa6c96483](https://user-images.githubusercontent.com/94169318/200604930-3c85e57d-baee-4bb9-9fba-8f1df0bc8527.png)
![196483526-9c31bde1-7680-4aa2-bf9c-50eae920a5ab](https://user-images.githubusercontent.com/94169318/200604986-08adc47a-5dfc-4e98-85bd-82afa250c273.png)
![196483562-c6a6db09-0b71-4a21-9a73-1717d2b4183d](https://user-images.githubusercontent.com/94169318/200605030-6466f5be-f599-41c7-bd38-8f87af990726.png)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

