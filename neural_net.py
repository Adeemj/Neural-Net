'''To do: 1)use public and private in class 
2)change J for more than one output
3)vectorize step in gradient evaluation
4)try on MNIST
5)initialize theta

'''

import numpy as np
import matplotlib.pyplot as plt

class Layer():
    def __init__(self,output_dim,input_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer = None
        self.output = None
        self.theta = None
        

def sigmoid(z):
    return 1/(1+np.exp(-z))

class Neural_net():
    
    def __init__(self):        
        self.layers = []
        self.J=[]        

    def read_full_theta(self,full_theta):
        self.full_theta = full_theta
        self.unroll_theta()
        
    def add(self,output_dim,input_dim=None):
        layer = Layer(output_dim,input_dim)
        #input_dim req if first layer after input layer
        if len(self.layers)!=0:
            layer.input_dim = self.layers[-1].output_dim
        self.layers.append(layer)

    def predict(self,X):
        
        input_matrix = X
        for layer in self.layers:
            m,n = input_matrix.shape
            #adding the bias unit
            ones = np.ones((1,n))
            input_matrix = np.concatenate((ones,input_matrix), axis=0)
            #calculating the output of this layer
            #output of this layer is the input of next layer
            input_matrix = sigmoid(layer.theta.dot(input_matrix))
            layer.output = input_matrix
            #return output of the final layer
        return input_matrix
        
    def unroll_theta(self):
        i = 0
        for layer in self.layers:
            input_dim,output_dim = layer.input_dim,layer.output_dim
            #adding one for bias unit
            length = output_dim*(input_dim+1)
            temp_vector =  self.full_theta[i:i+length]
            i +=length
            layer.theta = temp_vector.reshape(output_dim,input_dim+1)

    
    def train(self,X_train,y_train,lambda_param,alpha,iterations):
        self.X_train = X_train
        self.y_train = y_train
        self.lambda_param = lambda_param
        self.alpha = alpha
        self.iterations = iterations
        self.gradient_descent()
        plt.plot(range(len(self.J)),self.J)
    
    def cost(self):
        
        X_train = self.X_train
        y_train = self.y_train
        lambda_param = self.lambda_param
        
        m,n = X_train.shape
        y = np.transpose(y_train)
        X = np.transpose(X_train)
        predictions = self.predict(X)
        #calculating cost
        J = (-1/m)*np.sum(y*(np.log(predictions)) + (1-y)*np.log(1-predictions))

        self.J.append(J)
        #adding regularization to cost
        for layer in self.layers:
            J+=(lambda_param/(2*m))*np.sum((np.square(layer.theta))[:,1:])#bias parameter not regularized
        return J
    
    def gradient(self):
        J = self.cost()
        X_train = self.X_train
        y_train = self.y_train
        lambda_param = self.lambda_param
        
        m,n = X_train.shape
        y = np.transpose(y_train)
        X = np.transpose(X_train)
        predictions = self.predict(X)
        
        #computing deltas
        l = len(self.layers)
        
        delta=[None]*(l)
        delta[-1] = self.layers[-1].output-y
        lst = list(range(l-1))
        lst.reverse()
        for i in lst:
            theta_ = (self.layers[i+1].theta)[:,1:]
            layer_output = self.layers[i].output
            delta[i] = (np.transpose(theta_).dot(delta[i+1]))*layer_output*(1-layer_output)#

        Delta = [None]*(l)
        #try vectorising later
        for i in range(m):
            for j in range(l):
                if (i==0):
                    Delta[j] = np.zeros(((self.layers[j]).theta).shape)
                if (j==0):
                    output_of_prev_layer = X[:,[i]]
                else:    
                    output_of_prev_layer = self.layers[j-1].output[:,[i]]
                output_of_prev_layer = np.concatenate((np.ones((1,1)),output_of_prev_layer), axis=0)
                Delta[j] += (delta[j])[:,[i]].dot(np.transpose(output_of_prev_layer))
        
        gradient = np.zeros((self.full_theta).shape)
        k = 0
        for j in range(l):
            rows,cols = (self.layers[j].theta).shape
            length = rows*cols
            ones_col = np.ones((rows,1))
            Delta[j] = Delta[j]/m + (lambda_param)*np.concatenate((ones_col,(self.layers[j].theta)[:,1:]), axis=1)
            
            gradient[k:k+length] = Delta[j].reshape(length,1)
            k+=length
        return gradient
    
    def num_grad(self):
        perturb = 0.0001
        num_grad = np.zeros((self.full_theta).shape)
        for i in range(len(self.full_theta)):
            cost_i = self.cost()
            self.full_theta[i] += perturb
            cost_f = self.cost()
            self.full_theta[i] -= perturb
            num_grad[i] = (cost_f - cost_i)/perturb
        return num_grad
            
    
    def gradient_descent(self):
        alpha = self.alpha
        iterations = self.iterations
        for i in range(iterations):
            print(i)
            
            self.full_theta -= alpha*self.gradient()

#let's test
my_neural_net = Neural_net()
my_neural_net.add(input_dim=2,output_dim=2)
my_neural_net.add(output_dim=1)
theta = np.random.normal(0,1,9).reshape(9,1)
print(theta)
print('hi')
my_neural_net.read_full_theta(theta)

X_train = np.zeros((40,2))
X_train[:20,[0]] = np.random.normal(0,0.01,(20,1))
X_train[20:,[0]] = np.random.normal(1,0.01,(20,1))
X_train[:10,[1]] = np.random.normal(0,0.01,(10,1))
X_train[10:20,[1]] = np.random.normal(1,0.01,(10,1))
X_train[20:30,[1]] = np.random.normal(0,0.01,(10,1))
X_train[30:,[1]] = np.random.normal(1,0.01,(10,1))
y_train = np.zeros((40,1))
y_train[0:10] = np.ones((10,1))
y_train[30:] = np.ones((10,1))

my_neural_net.train(X_train,y_train,0,1,1000)
print(my_neural_net.full_theta)
inp = np.array([0,0,1,1,0,1,0,1]).reshape(2,4)
my_neural_net.predict(inp)
