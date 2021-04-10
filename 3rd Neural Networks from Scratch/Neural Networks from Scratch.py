import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import math
# first layer
# 4 inputs
inputs = [[1,2,3,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]]
# 3 neurons
# each neurons has 4 weights for 4 inputs

weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]

# each neurons has one bias value
biases = [2,3,0.5]

### without using matrix dot method to compute ###
'''layer_outputs = [] # output of current layers
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0 # output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output +=n_input*weight
    neuron_output +=neuron_bias
    layer_outputs.append(neuron_output)
# output = [inputs[0]*weights1[0]+inputs[1]*weights1[1]+inputs[2]*weights1[2]+inputs[3]*weights1[3]+bias1,
#           inputs[0]*weights2[0]+inputs[1]*weights2[1]+inputs[2]*weights2[2]+inputs[3]*weights2[3]+bias2,
#           inputs[0]*weights3[0]+inputs[1]*weights3[1]+inputs[2]*weights3[2]+inputs[3]*weights3[3]+bias3,]
print(layer_outputs)'''

### Adding one more layer ###
# second layer
# 3 inputs from last layer output
# 3 neurons
weights2 = [[0.1,-0.14,-0.5],
           [-0.5,0.12,-0.33,],
           [-0.44,0.73,-0.13]]
# 3 biases for 3 neurons
biases2 = [-1,2,-0.5]


layer1_outputs = np.dot(inputs,np.array(weights).T)+biases
layer2_outputs = np.dot(layer1_outputs,np.array(weights2).T)+biases2
print(layer2_outputs)

# np.random.seed(0)
X = [[1,2,3,2.5],
     [2.0,5.0,-1.0,2.0],
     [-1.5,2.7,3.3,-0.8]]



class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs,n_neurons) # don't need to transpose
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

layer1 = Layer_Dense(4,5)# 4 input, 5 output neurons
layer2 = Layer_Dense(5,2)# 5 input, 2 output neurons
layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
'''
Tutorial:
https://www.youtube.com/watch?v=gmjzbpSVY1A&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=5&ab_channel=sentdex
'''

inputs = [0,2,-1,3.3,-2.7,1.1,2.2,-100]
output = []
print(output)

for i in inputs:
        output.append(max(0,i))

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

nnfs.init()
X, y = spiral_data(100,3)
layer1 = Layer_Dense(2,5)# 2 input(features), 5 output neurons
layer1.forward(X)
activation1 = Activation_ReLU()
#print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)


'''
Tutorial:
https://www.youtube.com/watch?v=omz_NdFgWyU&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=6&ab_channel=sentdex
'''

layer_outputs = [[4.8,1.21,2.385],
                 [8.9,-1.81,0.2],
                 [1.41,1.051,0.026]]
#E = math.e

# Softmax Activation(input>expo>normalize>output)
expo_values = np.exp(layer_outputs)
norm_base = np.sum(expo_values, axis=1, keepdims=True)# sum each row
norm_values = expo_values / norm_base
print(norm_values)
# after do the normaliztion, the x will be in only 0-1, which can avoid overflow in Ex

class Activation_Softmax:
    def forward(self, inputs):
        expo_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # max value for each row
        probabilities = expo_values / np.sum(expo_values, axis=1, keepdims=True)
        self.output = probabilities

X, y = spiral_data(samples = 100,classes = 3)
dense1 = Layer_Dense(2,3)# 2 input(features), 3 output neurons
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()
dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])

'''
Tutorial:
https://www.youtube.com/watch?v=dEXPMQXoiLc&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=7&ab_channel=sentdex
'''
'''solving for x
e **x = b'''
b = 5.2
print(np.log(b))
print(np.exp(np.log(b)))
'''L = - sum(Yj log(Ypj))
one_hot = [1,0,0]
prediction = [0.7,0.1,0.2]

L = -(1*log(0.7)+0*log(0.1)+0*log(0.2)) = 0.35……'''

softmax_output = [0.7,0.1,0.2]
target_output = [1,0,0]

loss = -(math.log(softmax_output[0])*target_output[0]+
         math.log(softmax_output[1])*target_output[1]+
         math.log(softmax_output[2])*target_output[2])
print(loss)
loss = -math.log(softmax_output[0])
print(-math.log(0.7)) # higher confidence has lower loss
print(-math.log(0.5)) # lower confidence has higher loss



