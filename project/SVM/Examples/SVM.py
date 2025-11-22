import jax.numpy as np
from jax import jit
from functools import partial
import jax
#Toutes les fonctions sont Jitables directement normalement, seul doute sur fit mais vu que les arguments sont fixés 
#Le compiler XLA devrait bien pouvoir la traiter sans que je doive utiliser des partials arguments

class SVM:

    def __init__(self, C = 1.0):
        # C = error term
        self.C = C
        self.w = 0
        self.b = 0

    # Hinge Loss Function / Calculation
    @partial(jit, static_argnums=(0,))
    def hingeloss(self, w, b, x, y):
        # Regularizer term
        reg = 0.5 * (w * w)

        for i in range(x.shape[0]):
            # Optimization term
            opt_term = y[i] * ((np.dot(w, x[i])) + b)
            # my problem here is that i need to know if 1-opt term is > 0 or not but at compile time i only know the shape
            # calculating loss
            # problem here, i could use cond from jax control flow to replace max
   
            loss = reg + self.C * np.maximum(0, 1-opt_term)
        return loss[0][0]
    
    @partial(jit, static_argnums=(0,))
    def fit(self, X, Y, batch_size=100, learning_rate=0.001, epochs=1000):
        # The number of features in X
        number_of_features = X.shape[1]

        # The number of Samples in X
        number_of_samples = X.shape[0]

        c = self.C
        #used to compare the tracer object float 32(which contains one element) with 1
        comparator = np.array([1])

        # Creating ids from 0 to number_of_samples - 1
        ids = np.arange(number_of_samples)
        key = jax.random.key(0)
        # Shuffling the samples randomly
        jax.random.permutation(key,ids)

        # creating an array of zeros
        w = np.zeros((1, number_of_features))
        b = 0
        losses = []

        # Gradient Descent logic
        for i in range(epochs):
            # Calculating the Hinge Loss
            l = self.hingeloss(w, b, X, Y)

            # Appending all losses 
            losses.append(l)
            
            # Starting from 0 to the number of samples with batch_size as interval
            for batch_initial in range(0, number_of_samples, batch_size):
                gradw = 0
                gradb = 0

                for j in range(batch_initial, batch_initial+ batch_size):
                    if j < number_of_samples:
                        x = ids[j]
                        ti = Y[x] * (np.dot(w, X[x].T) + b)
                        type(ti)
                        
                       #Je veux convertir mon jit tracer float initial en un singleton booléen,
                       #Pour pouvoir utiliser any, dans ma jax cond et donc pouvoir imiter ce comportement
                        testing = np.greater(ti,comparator)

                        
                        print(c * Y[x] * X[x])
                        gradw += jax.lax.cond(np.any(testing), lambda _ : c * Y[x] * X[x], lambda _: [[0.0],[0.0]], None)
                        gradb += jax.lax.cond(np.any(testing), lambda _ : c * Y[x] * X[x], lambda _: [[0.0],[0.0]], None) 
                        # here there might be a trick to do, to not modify gradw && gradb when ti > 1
                        # else initial 
    
                # Updating weights and bias
                w = w - learning_rate * w + learning_rate * gradw
                b = b + learning_rate * gradb
        
        self.w = w
        self.b = b

        return self.w, self.b, losses
    
    @partial(jit, static_argnums=(0,))
    def predict(self, X):
        
        prediction = np.dot(X, self.w[0]) + self.b # w.x + b
        return np.sign(prediction)