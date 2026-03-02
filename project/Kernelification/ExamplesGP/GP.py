import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
from jax.tree_util import register_pytree_node_class
import optax
from functools import partial

@register_pytree_node_class
class GP:
    def __init__(self, Kernel, Alpha=1e-10, NormalizeObs=False):
        self.Kernel = Kernel
        self.Alpha = Alpha
        self.NormalizeObs = NormalizeObs
        self.TrainingData = None
        self.Obs = None
        self.L_ = None
        self.Alpha_ = None
        self.ObsMean = None
        self.ObsSTD = None


    #Tree flattening static instances as aux_data and dynamic as children
    def tree_flatten(self):
        children = (self.TrainingData, self.Obs, self.L_, self.Alpha_,
                    self.ObsMean, self.ObsSTD, self.Kernel)
        aux_data = {'Alpha': self.Alpha, 'NormalizeObs': self.NormalizeObs}
        return (children, aux_data)
    

    #Unflatenning to have a pytree(mostly for JAX operations)
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.__dict__.update(aux_data)
        (obj.TrainingData, obj.Obs, obj.L_, obj.Alpha_,
         obj.ObsMean, obj.ObsSTD, obj.Kernel) = children
        return obj

    @staticmethod
    #Marginal log likelihood that we will to maximize, to find the best hyperparameters values for our Kernels
    def marginal_log_likelihood(Kernel, X, Obs, Alpha):
        n = X.shape[0]
        K = Kernel(X)                      
        K = K + Alpha * jnp.eye(n)            
        L = jnp.linalg.cholesky(K)            
        #Resolution of L @ L.T @ Alpha_ = y
        Alpha_ = jax.scipy.linalg.cho_solve((L, True), Obs)
        #Log‑likelihood
        log_lik = -0.5 * jnp.sum(Obs * Alpha_) - jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * n * jnp.log(2 * jnp.pi)
        return log_lik

    def fit(self, X, Obs, num_iters=1000, learning_rate=0.01, progress=False):
      
        X = jnp.asarray(X)
        Obs = jnp.asarray(Obs)
        if Obs.ndim == 1:
            Obs = Obs[:, None]  #Converting to (n,1) if needed

        #Normalizing the data if needed, we make the hypothesis that the mean is null
        #, but if the mean isn't then we need to normalize the Obs

        if self.NormalizeObs:
            Obs_mean = jnp.mean(Obs, axis=0)
            Obs_std = jnp.std(Obs, axis=0)
            Obs_std = jnp.where(Obs_std == 0, 1.0, Obs_std)
            Obs_norm = (Obs - Obs_mean) / Obs_std
        else:
            Obs_mean = jnp.zeros(Obs.shape[1:])
            Obs_std = jnp.ones_like(Obs_mean)
            Obs_norm = Obs

        #Simple loss function (log likelihood that we will minimize)
        def loss_fn(Kernel):
            return -self.marginal_log_likelihood(Kernel, X, Obs_norm, self.Alpha)

        
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

        #Calculating the new values with optimized paramaters
        K = self.Kernel(X)
        K = K + self.Alpha * jnp.eye(len(X))
        L = jnp.linalg.cholesky(K)
        Alpha_ = jax.scipy.linalg.cho_solve((L, True), Obs_norm)

        #Updating the state
        self.TrainingData = X
        self.Obs = Obs_norm
        self.L_ = L
        self.Alpha_ = Alpha_
        self.ObsMean = Obs_mean
        self.ObsSTD = Obs_std

        return self

    @partial(jit, static_argnames=['Return_std', 'Return_cov'])
    def predict(self, X, Return_std=False, Return_cov=False):
     X = jnp.asarray(X)

     #Calculation of the diagonal(variance for each test random variable)
     Diag_fn = jax.vmap(lambda x: self.Kernel(x, x))

     if self.TrainingData is None:
        #If the model wasn't trained we calculate the prior(basically it's just applying,
        # The kernel for each pair of test points)
        n_targets = 1 if self.ObsMean is None else self.ObsMean.shape[0]
        Obs_mean = jnp.zeros((X.shape[0], n_targets)).squeeze()
        if Return_cov:
            Obs_cov = self.Kernel(X)
            return Obs_mean, Obs_cov
        elif Return_std:
            Obs_var = Diag_fn(X)   
            return Obs_mean, jnp.sqrt(Obs_var)
        else:
            return Obs_mean
     else:
        # If the model is trained then we apply the formulas to find the new distributions
        # so basically we want to find the updated mean and cov matrix
        K_trans = self.Kernel(X, self.TrainingData)
        Obs_mean = K_trans @ self.Alpha_
        Obs_mean = self.ObsSTD * Obs_mean + self.ObsMean
        if Obs_mean.ndim > 1 and Obs_mean.shape[1] == 1:
            Obs_mean = Obs_mean.squeeze(1)

        if Return_cov:
            v = jax.scipy.linalg.solve_triangular(self.L_, K_trans.T, lower=True)
            Obs_cov = self.Kernel(X) - v.T @ v
            Obs_cov = jnp.outer(Obs_cov, self.ObsSTD**2).reshape(*Obs_cov.shape, -1)
            if Obs_cov.shape[-1] == 1:
                Obs_cov = Obs_cov.squeeze(-1)
            return Obs_mean, Obs_cov
        elif Return_std:
            v = jax.scipy.linalg.solve_triangular(self.L_, K_trans.T, lower=True)
            Obs_var = Diag_fn(X) - jnp.einsum('ij,ji->i', v.T, v)
            Obs_var = Obs_var * self.ObsSTD**2
            return Obs_mean, jnp.sqrt(Obs_var)
        else:
            return Obs_mean