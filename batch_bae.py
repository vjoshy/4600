from  nnfs.datasets import spiral_data
import numpy as np
import nnfs
import matplotlib.pyplot as plt

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


    def forward(self, inputs, training):
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

class Layer_Dropout:

    def __init__(self, rate):
        self.rate = 1 - rate

    def forward (self, inputs, training):
        self.inputs = inputs

        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape)/ self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


# Input "layer"
class Layer_Input:
    # Forward pass
    def forward(self, inputs, training):
        self.output = inputs

class Activation_ReLU:

    def forward(self, inputs, training):
        self.inputs = inputs 
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

        # zero gradient where inputs were negative
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs


class Activation_Softmax:

    def forward(self, inputs, training):

        self.inputs = inputs

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

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
    


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

    def forward(self, inputs, training):

        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        return (outputs > 0.5) * 1


class Activation_Linear:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        return outputs

class Loss:

    # L1 and L2 regularization loss
    def reg_loss(self):

        reg_loss = 0

        for layer in self.trainable_layers:

            if layer.weight_regularizer_l1 > 0:
                reg_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

            if layer.weight_regularizer_l2 > 0:
                reg_loss += layer.weight_regularizer_l2 * np.sum((layer.weights)**2)

            if layer.bias_regularizer_l1 > 0:
                reg_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            if layer.bias_regularizer_l2 > 0:
                reg_loss += layer.bias_regularizer_l2 * np.sum((layer.biases)**2)

        return reg_loss
    
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers
    

    def calculate(self, output, y, *, include_reg = False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        # add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        
        if not include_reg:
            return data_loss

        return data_loss, self.reg_loss()
    
    # Calculates accumulated loss
    def calculate_accumulated(self, *, include_reg=False):
        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # If just data loss - return it
        if not include_reg:
            return data_loss
        
        # Return the data and regularization losses
        return data_loss, self.reg_loss()
    
    # Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


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

        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        sample_losses = - (y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses
    
    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1-1e-7)
        
        self.dinputs = -(y_true / clipped_dvalues -(1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs/samples


class Loss_MeanSquaredError(Loss): # L2 loss

    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2 , axis = -1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        outputs = len(dvalues[0]) # number of outputs in every sample

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


class Accuracy:

    def calculate(self, predictions, y):    

        # calculate accuracy for current predictions        
        # Get comparison results
        comparisons = self.compare(predictions, y)
        
        # Calculate accuracy
        accuracy = np.mean(comparisons)
        
        # Update accumulated values
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        
        return accuracy

    def calculate_accumulated(self):

        #Calculate accumulated accuracy over multiple batches/iterations
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy

    def new_pass(self):

        # Reset accumulated values for new pass/epoch
        self.accumulated_sum = 0
        self.accumulated_count = 0


class Accuracy_Categorical(Accuracy):    
    def init(self, y):
        pass

    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y


class Accuracy_Regression(Accuracy):
    
    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        
        if self.precision is None or reinit:
            self.precision = np.std(y) / 1000

    def compare(self, predictions, y):
      return np.absolute(predictions - y) < self.precision
    

# Model class
class Model:

    def __init__(self):
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)


    # Set loss, optimizer and accuracy
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    # Finalize the model
    def finalize(self):

        # Create and set the input layer
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)

        self.trainable_layers = []
        
        for i in range(layer_count):

            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
           
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
                
                # Update loss object with trainable layers
                self.loss.remember_trainable_layers(
                self.trainable_layers
                )
        
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CrossEntropy):        
            self.softmax_classifier_output = Activation_Softmax_Loss_CrossEntropy()

    # train model
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        self.accuracy.init(y)
        train_steps = 1
        
        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data

        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1
                
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        # main training loop
        for epoch in range(1, epochs+1):
            if not epoch % print_every:
                print(f'epoch: {epoch}')

            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                output = self.forward(batch_X, training=True)
                
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_reg=True)
                loss = data_loss + regularization_loss
                
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)
                
                self.backward(output, batch_y)
                
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not epoch % print_every and not batch_size:
                    print(f'step: {step}, acc: {accuracy:.3f}, loss: {loss:.3f} (data_loss: {data_loss:.3f}, reg_loss: {regularization_loss:.3f}), lr: {self.optimizer.current_learning_rate:.6f}')

            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_reg=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            
            if not epoch % print_every:
                print(f'training, acc: {epoch_accuracy:.3f}, loss: {epoch_loss:.3f} (data_loss: {epoch_data_loss:.3f}, reg_loss: {epoch_regularization_loss:.3f}), lr: {self.optimizer.current_learning_rate:.6f}')

            if validation_data is not None:
                self.loss.new_pass()
                self.accuracy.new_pass()

                for step in range(validation_steps):
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val
                    else:
                        batch_X = X_val[step*batch_size:(step+1)*batch_size]
                        batch_y = y_val[step*batch_size:(step+1)*batch_size]

                    output = self.forward(batch_X, training=False)
                    self.loss.calculate(output, batch_y)
                    predictions = self.output_layer_activation.predictions(output)
                    self.accuracy.calculate(predictions, batch_y)

                validation_loss = self.loss.calculate_accumulated()
                validation_accuracy = self.accuracy.calculate_accumulated()
                if not epoch % print_every:
                    print(f'validation, acc: {validation_accuracy:.3f}, loss: {validation_loss:.3f}')

    def forward(self, X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        
        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)


'''Training and testing'''

data = np.loadtxt('fashion-mnist_train.csv', delimiter=',', skiprows=1)
np.random.shuffle(data)

m,n = data.shape

train = data[100:2000]
test = data[0:100]
y = train[:,0].astype(int)
x = train[:, 1:]/255
x_test = test[:, 1:]/255

model = Model()

# Add layers
model.add(Layer_Dense(n - 1, 128, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.2))
model.add(Layer_Dense(128, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 128, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, n-1))
model.add(Activation_Sigmoid())

# Set loss, optimizer and accuracy
model.set(
    loss=Loss_BinaryCrossEntropy(),
    optimizer=Optimizer_Adam(learning_rate=0.001, decay=1e-5),
    accuracy=Accuracy_Regression()
)

# Finalize model
model.finalize()

# Train the model
model.train(x, x,  # Input same as target for autoencoder
           epochs=2001,
           batch_size=128,  # set batch
           print_every=100,
           validation_data=(x_test, x_test))

# Get reconstruction for test data
output = model.forward(x_test, training=False)
reconstructed = output

# Visualization function
def plot_reconstruction(x_original, x_reconstructed, index):
    plt.figure(figsize=(8, 4))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.gray()
    plt.imshow(x_original[index].reshape((28, 28)) * 255)
    plt.title('Original')
    plt.axis('off')
    
    # Reconstructed image
    plt.subplot(1, 2, 2)
    plt.gray()
    plt.imshow(x_reconstructed[index].reshape((28, 28)) * 255)
    plt.title('Reconstructed')
    plt.axis('off')
    
    plt.show()

# Show reconstructions
for i in range(3):  # Show 3 examples
    plot_reconstruction(x_test, reconstructed, i)