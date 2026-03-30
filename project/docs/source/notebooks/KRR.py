import numpy as np
from scipy import spatial as ssp
import matplotlib.pylab as plt
from kernax.stationary import Matern32Kernel
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
from jax.tree_util import register_pytree_node_class
import optax
from functools import partial
import jax

@register_pytree_node_class
class KRR:
    
    

    def __init__(self,X, Kernel=None,
                  RidgeCoeff = 1e-10, NormalizeObs=False, **kwds  ):
    
        #ok donc jv devoir calculer les coefficients duaux, et j'ai juste besoin de la mat 
        # de gram et l'inverse, jpeux direct faire cholesky ici, à voir
        #Je calcule pas les coefficients duaux ici prcq ça aurait aucun sens étant donné que je peux normaliser
        # mes cibles si besoin
        assert X.ndim == 2, "X must be 2D (number of samples, number of features)"
        assert RidgeCoeff >= 0, f"RidgeCoeff must be non-negative, got {RidgeCoeff}"
        if Kernel is None:
            self.Kernel=Matern32Kernel(length_scale=1.0)   
        else:
            self.Kernel = Kernel    
        self.TrainingData = jnp.asarray(X)
        self.RidgeCoeff = RidgeCoeff
        self.ObsMean = None
        self.ObsSTD = None
        self.DualCoefficients = None    
        self.NormalizeObs = NormalizeObs


    #This method use, is to transform my classes objects
    #into pytree, as jax need the 'self' arg to be a pytree to jit it 

    def tree_flatten(self):
        children = (self.TrainingData, self.Kernel)
        aux_data = {'RidgeCoeff': self.RidgeCoeff, 'NormalizeObs' : self.NormalizeObs}
        return (children, aux_data)
    
 
    #Unflatenning to have a pytree(mostly for JAX operations)
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.__dict__.update(aux_data)
        (obj.TrainingData, obj.Kernel) = children
        return obj

    @staticmethod
    #Marginal log likelihood that we will to maximize, to find the best hyperparameters values for our Kernels
    def marginal_log_likelihood(Kernel, X, Obs, RidgeCoeff):
        n = X.shape[0]
        K = Kernel(X)                      
        K = K + RidgeCoeff * jnp.eye(n)            
        L = jnp.linalg.cholesky(K)            
        #Resolution of the equation involving choleksy to find the dual coefficients
        Alpha_ = jax.scipy.linalg.cho_solve((L, True), Obs)
        #Loglikelihood
        log_lik = -0.5 * jnp.sum(Obs * Alpha_) - jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * n * jnp.log(2 * jnp.pi)
        return log_lik
    

    def fit(self, y, num_iters=1000, learning_rate=0.1, progress=False):
        ''' Fitting the images associated with our training data'''
        Obs = jnp.asarray(y)
    

        if Obs.ndim==1:
            Obs = Obs[:, None]

        if self.NormalizeObs:
            ObsMean = jnp.mean(Obs, axis=0)
            ObsSTD = jnp.std(Obs, axis=0)
            ObsSTD = jnp.where(ObsSTD == 0, 1.0,ObsSTD)
            ObsNorm = (Obs -ObsMean) /ObsSTD
        else:
           #The means and STD are calculated even if I don't normalize
           #So in predict I don't have to distingush
           #the normalized and unormalized cases
           ObsMean = jnp.zeros(Obs.shape[1:])
           ObsSTD = jnp.ones_like(ObsMean)
           ObsNorm = Obs
         #Simple loss function (log likelihood that we will minimize)
        def loss_fn(Kernel):
            return -self.marginal_log_likelihood(Kernel, self.TrainingData, ObsNorm, self.RidgeCoeff)

        
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(self.Kernel)

        #Classical optimization of the hyperparameters of the kernel
        @jit
        def step(Kernel, opt_state):
            loss, grads = value_and_grad(loss_fn)(Kernel)
            updates, opt_state = optimizer.update(grads, opt_state, Kernel)
            Kernel = optax.apply_updates(Kernel, updates)
            return Kernel, opt_state, loss

        #Optimization loop that give us losses
        Kernel = self.Kernel
        for i in range(num_iters):
            Kernel, opt_state, loss = step(Kernel, opt_state)
            if progress and i % 100 == 0:
                print(f"Iter {i}, loss = {loss:.4f}")

        #Update of the kernel optimized(the hyparmaters)
        self.Kernel = Kernel

        #Calculating the new values with optimized parameters
        K = self.Kernel(self.TrainingData,self.TrainingData)
        K = K + self.RidgeCoeff * jnp.eye(len(self.TrainingData))
        L = jnp.linalg.cholesky(K)
        DualCoefficients = jax.scipy.linalg.cho_solve((L, True), ObsNorm)
        
        self.ObsMean = ObsMean
        self.ObsSTD = ObsSTD
        self.DualCoefficients = DualCoefficients
        
        #We calculate the gram matrix after optimization of the hyperparameters
        GramMatrix = self.Kernel(self.TrainingData,self.TrainingData)    
        self.NormalizeObs = jnp.dot(GramMatrix,self.DualCoefficients)
        return self


    def predict(self,x):
        '''Prediction of y for feature space points
        
        '''
        x = jnp.asarray(x)
        distxpredict = self.Kernel(x, self.TrainingData)
        yPredictionNorm = jnp.dot(distxpredict, self.DualCoefficients)
        yPredictionUnnorm = yPredictionNorm * self.ObsSTD + self.ObsMean
        return yPredictionUnnorm




