import jax.numpy as jnp
import sys
sys.path.insert(0,"2526_INFOB318_Kernax/project/SVM")
from jax import jit
from functools import partial
from kernax import LinearKernel, SEKernel
import jax
import optax

#Toutes les fonctions sont Jitables directement normalement, seul doute sur fit mais vu que les arguments sont fixés 
#Le compiler XLA devrait bien pouvoir la traiter sans que je doive utiliser des partials arguments

class SVM:

    def __init__(self, C = 1.0, kernel = None):
        # C = error term
        self.C = C
        self.w = 0
        self.b = 0
        self.kernel = None
        self.X_train = None  
        self.y_train = None  
        self.alpha = None
        self.kernel_fn = kernel





    
    # Hinge Loss Function / Calculation
    @partial(jit, static_argnums=(0,))
    def hingeloss(self, w, b, x, y):
    # Regularizer term
     reg = 0.5 * jnp.sum(w * w)  
    
    
    # We calculate all the scores in one time, so we don't use a for loop
     scores = jnp.dot(x, w) + b
     
    # hinge loss for all elements of our sample
     hinge_losses = jnp.maximum(0, 1 - y * scores)
    
    # mean of all elements of our sample
     avg_hinge = jnp.mean(hinge_losses)
    
    # total loss
     total_loss = reg + self.C * avg_hinge
    
     return total_loss
    
    

    @partial(jit, static_argnums=(0,))
    def dual_loss(self, alpha, X, y, K):
        
        n = len(y)
        
        # quadratic function for dual problem
        y_matrix = jnp.outer(y, y)
        quad_term = 0.5 * jnp.dot(alpha, jnp.dot(y_matrix * K, alpha))
        
        
        linear_term = -jnp.sum(alpha)
        
        
        penalty = self.C * jnp.sum(jnp.maximum(0, alpha - self.C)**2) + \
                  self.C * jnp.sum(jnp.maximum(0, -alpha)**2)
        
        return quad_term + linear_term + penalty




    def fit(self, X, Y, batch_size=100, learning_rate=0.001, epochs=1000):
        # The number of features in X
        self.X_train = X.copy()
        self.y_train = Y.copy()
        X = jnp.array(X, dtype=jnp.float32)
        Y = jnp.array(Y, dtype=jnp.float32)
        assert X.ndim == 2, "X should be in two dimensions"
        assert Y.ndim == 1, "Y should be in one dimension"
        assert len(X) == len(Y), "X and Y should have the same length"
    
        #As the dual problem and primal don't have have the same parameters, i need to optimize in two different ways
        if(self.kernel_fn is not None):
            K = self.kernel_fn(X, X) 
            # We initialize alphas (so basically that's the dual coefficients)
            n_samples = X.shape[0]
            alpha = jnp.zeros(n_samples)

            # We want to optimize our alphas

            optimizer = optax.adam(learning_rate=learning_rate)
            opt_state = optimizer.init(alpha)
            losses = []
            def body(i, state):
               alpha, opt_state = state
               loss, grad_alpha = jax.value_and_grad(self.dual_loss)(alpha, X, Y, K)
               updates, opt_state = optimizer.update(grad_alpha, opt_state, alpha)
               alpha = optax.apply_updates(alpha, updates)
               alpha = jnp.clip(alpha, 0, self.C)
               return alpha, opt_state

            alpha, opt_state = jax.lax.fori_loop(0, epochs, body, (alpha, opt_state))

            self.alpha = alpha    
            # When we are in the dual problem there's no weight
            self.w = None

            SvIndices = jnp.where(alpha > 1e-5)[0]  # support vectors
            if len(SvIndices) > 0:
                # for each support vector we need to calculate the bias
                b_vals = []
                for index in SvIndices:
                    prediction = jnp.sum(alpha[SvIndices] * Y[SvIndices] * K[SvIndices, index])
                    b_vals.append(Y[index] - prediction)
                self.b = jnp.mean(jnp.array(b_vals))
            else:
                self.b = 0
            
            return self.alpha, self.b, losses

        number_of_features = X.shape[1]

        # The number of Samples in X
        number_of_samples = X.shape[0]

        c = self.C

        # Creating Set from 0 to number_of_samples - 1
        Set = jnp.arange(number_of_samples)
        key = jax.random.key(0)
        # Taking values from the samples randomly
        jax.random.permutation(key,Set)

        # creating an array of zeros
        w = jnp.zeros(number_of_features)  
        b = 0.0
        losses = []


        # The paramaters that we're going to use for optax
        params = {'w': w, 'b': b}
        # The optimizer that we will use to update our w and b and find the best ones.
        optimizer = optax.adam(learning_rate=learning_rate)

        opt_state = optimizer.init(params)

        def body(i, state):
            params, opt_state = state
            loss, (grad_w, grad_b) = jax.value_and_grad(self.hingeloss, (0, 1))(params["w"], params["b"], X, Y)
            grads = {'w': grad_w, 'b': grad_b}
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        params, opt_state = jax.lax.fori_loop(0, epochs, body, (params, opt_state))
        self.w = params["w"]
        self.b = params["b"]
        return params["w"], params["b"], losses
            
    
    def predict(self, X):
        if(self.kernel_fn is not None):
            assert self.alpha is not None, "if there is no alphas then the model wasn't trained "
            assert self.X_train is not None, "no data points"
            
            # Kernel matrix between X and the training points
            K_test = self.kernel_fn(X, self.X_train)
            
            # decision function of dual problems
            decisions = jnp.dot(K_test, self.alpha * self.y_train) + self.b
            return jnp.sign(decisions)

            

        prediction = jnp.dot(X, self.w) + self.b
        return jnp.sign(prediction)