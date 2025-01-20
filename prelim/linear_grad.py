import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# initial parameters
N = 100
d = 3
W = np.array([5, 10 , 15])
b = 20
num_episodes = 10000

# some data
x = np.random.randn(N,d)
y = np.dot(W.T, x.T) + b

# learning rate
alpha = 0.001

m = np.zeros(W.shape) # slope
b = 0.0     # intercept

losses = []
epochs_record = []

for ep in range(num_episodes):
    dldm = 0
    dldb = 0
    N = x.shape[0]

    y_hat = np.dot(m.T, x.T) +  b

    # loss function or mse 
    loss = np.mean((y - y_hat) **2 )

    # SS_res = (y - yhat)**2 = (y - (mx + b))**2
   
    y_hat = np.dot(m.T, x.T) +  b

    # gradients
    dldm += np.dot(-2*x.T, (y - y_hat))
    dldb += -2*np.sum(y - y_hat)

    # update rule
    m = m - alpha * (1/N) * dldm
    b = b - alpha * (1/N) * dldb

    if ep%100 == 0:
        losses.append(loss)
        epochs_record.append(ep)


loss_df = pd.DataFrame({'Epoch': epochs_record, 'Cost': losses})

#plt.figure(figsize=(8, 5))
plt.scatter(loss_df['Epoch'], loss_df['Cost'], color='blue')
plt.plot(loss_df['Epoch'], loss_df['Cost'], color='red', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Cost (MSE)')
plt.title('Training Loss Over Time')
plt.grid(True)
plt.show()

print(f'm: {m}, b: {b}')

