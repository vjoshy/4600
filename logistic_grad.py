import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd

# some data

iris = datasets.load_iris()
X=iris.data[0:99,:2]
y=iris.target[0:99]# Plot the training points

plt.figure(2, figsize=(8, 6))

alpha = 0.01
num_episodes = 10000
N = len(X)
w = np.zeros((2,1))

b = 0

losses = []

for ep in range(num_episodes):
    # sigmoid function => predicted y
    z = np.dot(w.T, X.T) + b
    y_pred = 1/(1 + (1/np.exp(z)))

    # loss function
    loss = (-1/N) * np.sum(y * np.log(y_pred) + (1-y)* np.log(1-y_pred))

    # gradient
    dw = (1/N) * np.dot(X.T, (y_pred - y).T)
    db = (1/N) * np.sum(y_pred - y)

    # update w & b 
    w = w - alpha * dw
    b = b - alpha * db

    #Records cost
    if ep%1000 == 0:
        losses.append(loss)
        print(loss)


episodes = pd.DataFrame(list(range(100,num_episodes,100)))
loss_df = pd.DataFrame(losses)
loss_df=pd.concat([episodes, loss_df], axis=1)
loss_df.columns=['Epoch','Cost']
plt.scatter(loss_df['Epoch'], loss_df['Cost'])
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()