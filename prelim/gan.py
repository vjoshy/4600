from  nnfs.datasets import spiral_data
import numpy as np
import nnfs
import matplotlib.pyplot as plt
import math

nnfs.init()

class Layer_Dense:
    
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0, init_weight = 0.01):
        
        self.weights = init_weight * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2


    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        

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
    def forward(self, y_pred, y_true):
        # Ensure input shape is correct
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
            
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + 
                         (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)  # Use axis=-1 instead of 1
        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        
        # check correct shapes
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        if len(dvalues.shape) == 1:
            dvalues = dvalues.reshape(-1, 1)
            
        clipped_dvalues = np.clip(dvalues, 1e-7, 1-1e-7)
        
        # gradient
        self.dinputs = -(y_true / clipped_dvalues - 
                        (1 - y_true) / (1 - clipped_dvalues))
        
        # Normalize 
        self.dinputs = self.dinputs / samples
        

class Loss_MeanSquaredError(Loss):

    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2 , axis = -1)
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

# Adam optimizer
class Optimizer_Adam:
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums +  (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums /  (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache +  (1 - self.beta_2) * layer.dweights**2

        layer.bias_cache = self.beta_2 * layer.bias_cache +  (1 - self.beta_2) * layer.dbiases**2
        
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

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

import numpy as np

# Data preparation
train_data_length = 1024
batch_size = 64  # Added batch size for training

# Real data
train_data = np.zeros((train_data_length, 2))
train_data[:, 0] = 2 * np.pi * np.random.rand(train_data_length)
train_data[:, 1] = np.sin(train_data[:, 0])

# Discriminator
dense1 = Layer_Dense(2, 256, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation1 = Activation_ReLU()
dropout1 = Layer_Dropout(0.3)  

dense2 = Layer_Dense(256, 128,  weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation2 = Activation_ReLU()
dropout2 = Layer_Dropout(0.3)

dense3 = Layer_Dense(128, 64,  weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
activation3 = Activation_ReLU()
dropout3 = Layer_Dropout(0.3)

dense4 = Layer_Dense(64, 1)
activation4 = Activation_Sigmoid()

loss_function = Loss_BinaryCrossEntropy()
d_optimizer = Optimizer_Adam(learning_rate=0.001, decay=5e-5)

# Generator
gen_dense1 = Layer_Dense(2, 16, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
gen_activation1 = Activation_ReLU()

gen_dense2 = Layer_Dense(16, 32)
gen_activation2 = Activation_ReLU()

gen_dense3 = Layer_Dense(32, 2)
gen_loss = Loss_BinaryCrossEntropy()
gen_optimizer = Optimizer_Adam(learning_rate=0.001, decay=5e-5)

# Training loop
episodes = 20001

d_df = []
g_df = []

import matplotlib.animation as animation

def create_training_animation(num_frames=10, num_samples=1000):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        # Generate samples
        latent_space_samples = np.random.randn(num_samples, 2)
        gen_dense1.forward(latent_space_samples)
        gen_activation1.forward(gen_dense1.output)
        gen_dense2.forward(gen_activation1.output)
        gen_activation2.forward(gen_dense2.output)
        gen_dense3.forward(gen_activation2.output)
        generated_samples = gen_dense3.output
        
        # Plot real data
        ax1.scatter(train_data[:, 0], train_data[:, 1], c='blue', alpha=0.5, label='Real')
        ax1.set_title('Real Data')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        
        # Plot generated data
        ax2.scatter(generated_samples[:, 0], generated_samples[:, 1], c='red', alpha=0.5, label='Generated')
        ax2.set_title(f'Generated Data (Frame {frame})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend()
        
        plt.tight_layout()
    
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=10)
    plt.show()


for ep in range(episodes):

    # Train Discriminator
    # create real samples
    real_samples = train_data[np.random.randint(0, train_data_length, batch_size)]
    real_labels = np.ones((batch_size, 1))
    
    # create fake samples
    latent_space_samples = np.random.randn(batch_size, 2)
    
    # fake samples generation
    gen_dense1.forward(latent_space_samples)
    gen_activation1.forward(gen_dense1.output)
    gen_dense2.forward(gen_activation1.output)
    gen_activation2.forward(gen_dense2.output)
    gen_dense3.forward(gen_activation2.output)

    fake_samples = gen_dense3.output
    fake_labels = np.zeros((batch_size, 1))
    
    # Combine real and fake samples
    all_samples = np.vstack((real_samples, fake_samples))
    all_labels = np.vstack((real_labels, fake_labels))
    
    # Train discriminator
    dense1.forward(all_samples)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    
    dense2.forward(dropout1.output)
    activation2.forward(dense2.output)
    dropout2.forward(activation2.output)
    
    dense3.forward(dropout2.output)
    activation3.forward(dense3.output)
    dropout3.forward(activation3.output)
    
    dense4.forward(dropout3.output)
    activation4.forward(dense4.output)
    
    # Calculate discriminator loss
    d_loss_fxn = loss_function.calculate(activation4.output, all_labels)
    d_reg_loss = loss_function.reg_loss(dense1) + loss_function.reg_loss(dense2) + loss_function.reg_loss(dense3) + loss_function.reg_loss(dense4) 
    d_loss = d_loss_fxn + d_reg_loss
    
    # Discriminator backward pass
    loss_function.backward(activation4.output, all_labels)
    activation4.backward(loss_function.dinputs)
    dense4.backward(activation4.dinputs)
    
    dropout3.backward(dense4.dinputs)
    activation3.backward(dropout3.dinputs)
    dense3.backward(activation3.dinputs)
    
    dropout2.backward(dense3.dinputs)
    activation2.backward(dropout2.dinputs)
    dense2.backward(activation2.dinputs)
    
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)
    
    # Update discriminator
    d_optimizer.pre_update_params()
    d_optimizer.update_params(dense1)
    d_optimizer.update_params(dense2)
    d_optimizer.update_params(dense3)
    d_optimizer.update_params(dense4)
    d_optimizer.post_update_params()
    
    # Train Generator
    latent_space_samples = np.random.randn(batch_size, 2)
    
    # Generator forward pass
    gen_dense1.forward(latent_space_samples)
    gen_activation1.forward(gen_dense1.output)
    gen_dense2.forward(gen_activation1.output)
    gen_activation2.forward(gen_dense2.output)
    gen_dense3.forward(gen_activation2.output)
    
    # Pass generated samples through discriminator
    dense1.forward(gen_dense3.output)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)
    activation2.forward(dense2.output)
    dropout2.forward(activation2.output)
    dense3.forward(dropout2.output)
    activation3.forward(dense3.output)
    dropout3.forward(activation3.output)
    dense4.forward(dropout3.output)
    activation4.forward(dense4.output)
    
    # Calculate generator loss (wants discriminator to predict 1s)
    g_loss_fxn = loss_function.calculate(activation4.output, real_labels)
    g_reg_loss = loss_function.reg_loss(gen_dense1) + loss_function.reg_loss(gen_dense2) + loss_function.reg_loss(gen_dense3) 
    g_loss = g_loss_fxn + g_reg_loss

    # Generator backward pass
    loss_function.backward(activation4.output, real_labels)
    
    # Propagate back through discriminator
    activation4.backward(loss_function.dinputs)
    dense4.backward(activation4.dinputs)
    dropout3.backward(dense4.dinputs)
    activation3.backward(dropout3.dinputs)
    dense3.backward(activation3.dinputs)
    dropout2.backward(dense3.dinputs)
    activation2.backward(dropout2.dinputs)
    dense2.backward(activation2.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)
    
    # Propagate back through generator
    gen_dense3.backward(dense1.dinputs)
    gen_activation2.backward(gen_dense3.dinputs)
    gen_dense2.backward(gen_activation2.dinputs)
    gen_activation1.backward(gen_dense2.dinputs)
    gen_dense1.backward(gen_activation1.dinputs)
    
    # Update generator
    gen_optimizer.pre_update_params()
    gen_optimizer.update_params(gen_dense1)
    gen_optimizer.update_params(gen_dense2)
    gen_optimizer.update_params(gen_dense3)
    gen_optimizer.post_update_params()

    d_df.append(d_loss)
    g_df.append(g_loss)
    
    # Print progress
    if ep % 100 == 0:
        print(f"Epoch: {ep}, Discriminator Loss: {d_loss:.3f}, Generator Loss: {g_loss:.3f}")

ep_range = np.arange(0, episodes, 1) 
plt.figure(figsize=(10, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(ep_range, d_df, label='Discriminant Loss')
plt.xlabel('episodes')
plt.ylabel('Loss')
plt.title('Discriminant Loss over Time')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ep_range, g_df, label='Generator Loss')
plt.xlabel('episodes')
plt.ylabel('Loss')
plt.title('Generator Loss over Time')
plt.legend()

plt.show()


create_training_animation()