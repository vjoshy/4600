import numpy as np

x = np.array([1, 2, 3, 4])
y = np.array([2, 4, 6, 8])

w1 = np.array([[0.4, 0.6, 0.1, -0.3],
              [0.2, 0.8, -0.5, 1]])

b1 = np.array([0.5,  0.2])
o1 = np.dot(x, w1.T) + b1

print(o1)


w2 = np.array([[-0.2, 0.01], [0.1, 1]])
b2 = np.array([0.3,  0.1])

o2 = np.dot(o1, w2.T) + b2

print(o2)


w3 = np.array([[0.5, 0.2, 0.7, 0.1],
      [0.3, 0.6, 0.4, 0.8]])

b3 = [0.1, 0.1, 0.1, 0.1]

o3 = np.dot(o2, w3) + b3

print(o3)