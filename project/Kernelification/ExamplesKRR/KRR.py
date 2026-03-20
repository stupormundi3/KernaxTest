import numpy as np
from scipy import spatial as ssp
import matplotlib.pylab as plt
from kernax.stationary import Matern32Kernel
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
from jax.tree_util import register_pytree_node_class
import optax
from functools import partial

@register_pytree_node_class
class GaussProcess:
    
    

    def __init__(self,X, Kernel=None, y=None, 
                  RidgeCoeff = 1e-10, **kwds ):
    
        #ok donc jv devoir calculer les coefficients duaux, et j'ai juste besoin de la mat 
        # de gram et l'inverse, jpeux direct faire cholesky ici, à voir
        if Kernel is None:
            self.Kernel=Matern32Kernel(length_scale=1.0)   
        else:
            self.Kernel = Kernel    
        self.TrainingData = jnp.asarray(X)
        self.RidgeCoeff = RidgeCoeff
        self.GramMatrix = self.Kernel(self.TrainingData,self.TrainingData)
        self.Kinv = jnp.linalg.inv(self.GramMatrix +
                             jnp.eye(*self.GramMatrix.shape)*RidgeCoeff)
        if y is not None:
            self.y = y
            self.yest = self.fit(y)

    #This method use, is to transform my classes objects
    #into pytree, as jax need the 'self' arg to be a pytree to jit it 

    def tree_flatten(self):
        children = (self.TrainingData, self.Kernel, self.GramMatrix, self.Kinv,
                    self.y)
        aux_data = {'RidgeCoeff': self.RidgeCoeff, }
        return (children, aux_data)
    
 

    def fit(self,y):
        ''' Fitting the images associated with our training data'''
        self.DualCoefficients = jnp.dot(self.Kinv, y)
        TrainingDataFitted = jnp.dot(self.GramMatrix,self.DualCoefficients)
        return TrainingDataFitted


    def predict(self,x):
        '''Prediction of y for feature space points'''
        x = jnp.asarray(x)
        distxpredict = self.Kernel(x, self.TrainingData)
        return jnp.dot(distxpredict, self.DualCoefficients)

    def plot(self, y, plt=plt ):
        '''some basic plots'''
        #todo return proper graph handles
        plt.figure()
        plt.plot(self.x,self.y, 'bo-', self.x, self.yest, 'r.-')
        plt.title('sample (training) points')
        plt.figure()
        plt.plot(self.xpredict,y,'bo-',self.xpredict,self.ypredict,'r.-')
        plt.title('all points')



def example1():
    m,k = 500,4
    upper = 6
    xs1a = jnp.linspace(1,upper,m)[:,jnp.newaxis]
    xs1 = xs1a*jnp.ones((1,4)) + 1/(1.0+jnp.exp(np.random.randn(m,k)))
    xs1 /= jnp.std(xs1[::k,:],0)   # normalize scale, could use cov to normalize
    y1true = jnp.sum(jnp.sin(xs1)+jnp.sqrt(xs1),1)[:,jnp.newaxis]
    y1 = y1true + 0.250 * np.random.randn(m,1)

    stride = 2 #use only some points as trainig points e.g 2 means every 2nd
    gp1 = GaussProcess(xs1[::stride,:],y=y1[::stride,:], 
                       RidgeCoeff=1e-10)
    TrainingDataFittedr1 = gp1.predict(xs1)
    plt.figure()
    plt.plot(y1true, y1,'bo',y1true, TrainingDataFittedr1,'r.')
    plt.title('euclid kernel: true y versus noisy y and estimated y')
    plt.figure()
    plt.plot(y1,'bo-',y1true,'go-',TrainingDataFittedr1,'r.-')
    plt.title('euclid kernel: true (green), noisy (blue) and estimated (red) '+
              'observations')

    gp2 = GaussProcess(xs1[::stride,:],y=y1[::stride,:],
                       RidgeCoeff=1e-1)
    TrainingDataFittedr2 = gp2.predict(xs1)
    plt.figure()
    plt.plot(y1true, y1,'bo',y1true, TrainingDataFittedr2,'r.')
    plt.title('rbf kernel: true versus noisy (blue) and estimated (red) observations')
    plt.figure()
    plt.plot(y1,'bo-',y1true,'go-',TrainingDataFittedr2,'r.-')
    plt.title('rbf kernel: true (green), noisy (blue) and estimated (red) '+
              'observations')
    #gp2.plot(y1)


def example2(m=100, stride=2):
    #m,k = 100,1
    upper = 6
    xs1 = jnp.linspace(1,upper,m)[:,jnp.newaxis]
    y1true = jnp.sum(jnp.sin(xs1**2),1)[:,jnp.newaxis]/xs1
    y1 = y1true + 0.05*np.random.randn(m,1)

    RidgeCoeff = 1e-10
    #stride = 2 #use only some points as trainig points e.g 2 means every 2nd
    gp1 = GaussProcess(xs1[::stride,:],None, y=y1[::stride,:],
                       RidgeCoeff=1e-10)
    TrainingDataFittedr1 = gp1.predict(xs1)
    plt.figure()
    plt.plot(y1true, y1,'bo',y1true, TrainingDataFittedr1,'r.')
    plt.title('euclid kernel: true versus noisy (blue) and estimated (red) observations')
    plt.figure()
    plt.plot(y1,'bo-',y1true,'go-',TrainingDataFittedr1,'r.-')
    plt.title('euclid kernel: true (green), noisy (blue) and estimated (red) '+
              'observations')

    gp2 = GaussProcess(xs1[::stride,:],y=y1[::stride,:], 
                       RidgeCoeff=1e-2)
    TrainingDataFittedr2 = gp2.predict(xs1)
    plt.figure()
    plt.plot(y1true, y1,'bo',y1true, TrainingDataFittedr2,'r.')
    plt.title('rbf kernel: true versus noisy (blue) and estimated (red) observations')
    plt.figure()
    plt.plot(y1,'bo-',y1true,'go-',TrainingDataFittedr2,'r.-')
    plt.title('rbf kernel: true (green), noisy (blue) and estimated (red) '+
              'observations')
    #gp2.plot(y1)

if __name__ == '__main__':
    plt.close('all')
    example2()
    example1()
    plt.show()
