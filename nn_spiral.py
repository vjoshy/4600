from  nnfs.datasets import spiral_data
import numpy as np
import nnfs
import matplotlib.pyplot as plt

nnfs.init()


class Layer_Dense:
    
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2


    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    def backward(self, dvalues):
        # gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims=True)
        
        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases


        #gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs 
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

        # zero gradient where inputs were negative
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims= True))
        prob = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
        self.output = prob

    def backward(self, dvalues):

        # initiliaze array
        self.dinputs = np.empty_like(dvalues)

        # enumerate outputs and gradients
        for idx, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1) # flatten array

            # calculate jacobian matrix and sample-wise gradient (annd add to sample gradients)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[idx] = np.dot(jacobian, single_dvalues)

class Activation_Softmax_Loss_CrossEntropy:

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1 # calculate gradient
        self.dinputs = self.dinputs / samples # normalize gradient

class Activation_Sigmoid:

    def forward(self, inputs):

        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output


class Activation_Linear:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

class Loss:

    # L1 and L2 regularization loss
    def reg_loss(self, layer):

        reg_loss = 0

        if layer.weight_regularizer_l1 > 0:
            reg_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

        if layer.weight_regularizer_l2 > 0:
            reg_loss += layer.weight_regularizer_l2 * np.sum((layer.weights)**2)

        if layer.bias_regularizer_l1 > 0:
            reg_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

        if layer.bias_regularizer_l2 > 0:
            reg_loss += layer.bias_regularizer_l2 * np.sum((layer.biases)**2)

        return reg_loss

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        return data_loss


class Loss_CrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)

        neg_log_likelihood = -np.log(correct_confidences)

        return neg_log_likelihood
    
    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true/dvalues # calculate gradient
        self.dinputs = self.dinputs/samples #normalize gradient


class Loss_BinaryCrossEntropy(Loss):

    def forward(sef, y_pred, y_true):

        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        sample_losses = - (y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=1)

        return sample_losses
    
    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1-1e-7)
        
        self.dinputs = -(y_true / clipped_dvalues -(1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs/samples


class Loss_MeanSquaredError(Loss):

    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2 , axies = -1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        # gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples # normalize gradient

# Mean Absolute Error loss
class Loss_MeanAbsoluteError(Loss): # L1 loss

    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        outputs = len(dvalues[0])

        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples    # normalize gradient


class Optimizer_SGD:

    def __init__(self, learning_rate=1, decay=0., momentum = 0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def update_params(self, layer):

        if self.momentum:

            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)

                layer.bias_momentums = np.zeros_like(layer.biases)

            
            weight_udpates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_udpates

            bias_udpates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_udpates
        
        else:
        # subtracting (or -ve) to perfrom gradient descent 
        # if i used += instead, I'd be doing gradient ascent haha
            weight_udpates = -self.current_learning_rate * layer.dweights
            bias_udpates =  -self.current_learning_rate * layer.dbiases 

        layer.weights += weight_udpates
        layer.biases += bias_udpates

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


class Layer_Dropout:

    def __init__(self, rate):
        self.rate = 1 - rate

    def forward (self, inputs):
        self.inputs = inputs

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape)/ self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

'''Training and testing'''

x, y= spiral_data(samples=1000, classes=3)

accuracy_df = []
loss_df = []

dense1 = Layer_Dense(2,64, weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4)
activation1 = Activation_ReLU()

# dropout layer
dropout1 = Layer_Dropout(0.1)

dense2 = Layer_Dense(64,3)

loss_activation = Activation_Softmax_Loss_CrossEntropy()

optimizer = Optimizer_SGD(decay=1e-2, momentum=0.95)

episodes = 10001

for ep in range(episodes):

    # layer 1
    dense1.forward(x)
    activation1.forward(dense1.output)

    # layer 2
    dropout1.forward(activation1.output)
    
    dense2.forward(dropout1.output)
    data_loss = loss_activation.forward(dense2.output, y)

    regularization_loss = loss_activation.loss.reg_loss(dense1) + loss_activation.loss.reg_loss(dense2)

    loss = data_loss + regularization_loss

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if not ep % 100:
        print(f'ep: {ep}, 'f'acc: {accuracy:.3f}, 'f'loss: {loss:.3f}')
        accuracy_df.append(accuracy)
        loss_df.append(loss)
    

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)

    dropout1.backward(dense2.dinputs)

    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)



    # Update weights and biases
    optimizer.pre_update_params()

    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

    optimizer.post_update_params()

ep_range = np.arange(0, episodes, 100) 
plt.figure(figsize=(10, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(ep_range, accuracy_df, label='accuracy')
plt.xlabel('episodes')
plt.ylabel('Accuracy')
plt.title('Accuracy over Time')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ep_range, loss_df, label='loss')
plt.xlabel('episodes')
plt.ylabel('Loss')
plt.title('Loss over Time')
plt.legend()

plt.show()


# Create test dataset
X_test, y_test = spiral_data(samples=100, classes=3)
# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y_test)
# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)

accuracy = np.mean(predictions==y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')