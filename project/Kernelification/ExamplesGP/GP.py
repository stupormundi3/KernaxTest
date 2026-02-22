import jax
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
from jax.tree_util import register_pytree_node_class
import optax
from functools import partial

@register_pytree_node_class
class GPR:
    def __init__(self, kernel, alpha=1e-10, normalize_y=False):
        self.kernel = kernel
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.X_train_ = None
        self.y_train_ = None
        self.L_ = None
        self.alpha_ = None
        self._y_mean = None
        self._y_std = None

    def tree_flatten(self):
        children = (self.X_train_, self.y_train_, self.L_, self.alpha_,
                    self._y_mean, self._y_std, self.kernel)
        aux_data = {'alpha': self.alpha, 'normalize_y': self.normalize_y}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.__dict__.update(aux_data)
        (obj.X_train_, obj.y_train_, obj.L_, obj.alpha_,
         obj._y_mean, obj._y_std, obj.kernel) = children
        return obj

    @staticmethod
    def marginal_log_likelihood(kernel, X, y, alpha):
        n = X.shape[0]
        K = kernel(X)                        # matrice de Gram
        K = K + alpha * jnp.eye(n)            # ajout du bruit
        L = jnp.linalg.cholesky(K)            # factorisation de Cholesky
        # Résolution de L @ L.T @ alpha_ = y
        alpha_ = jax.scipy.linalg.cho_solve((L, True), y)
        # Log‑vraisemblance : -0.5 yᵀ α - ∑ log(L_ii) - n/2 log(2π)
        log_lik = -0.5 * jnp.dot(y.T, alpha_).squeeze() - jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * n * jnp.log(2 * jnp.pi)
        return log_lik

    def fit(self, X, y, num_iters=1000, learning_rate=0.01, verbose=False):
        # Convertir en JAX arrays
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        if y.ndim == 1:
            y = y[:, None]  # mettre sous forme (n,1)

        # Normalisation si demandée
        if self.normalize_y:
            y_mean = jnp.mean(y, axis=0)
            y_std = jnp.std(y, axis=0)
            y_std = jnp.where(y_std == 0, 1.0, y_std)
            y_norm = (y - y_mean) / y_std
        else:
            y_mean = jnp.zeros(y.shape[1:])
            y_std = jnp.ones_like(y_mean)
            y_norm = y

        # Fonction de perte (négative log‑vraisemblance) en fonction du kernel
        def loss_fn(kernel):
            return -self.marginal_log_likelihood(kernel, X, y_norm, self.alpha)

        # Initialiser l'optimiseur Adam
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(self.kernel)

        # Étape d'optimisation jittée
        @jit
        def step(kernel, opt_state):
            loss, grads = value_and_grad(loss_fn)(kernel)
            updates, opt_state = optimizer.update(grads, opt_state, kernel)
            kernel = optax.apply_updates(kernel, updates)
            return kernel, opt_state, loss

        # Boucle d'optimisation
        kernel = self.kernel
        for i in range(num_iters):
            kernel, opt_state, loss = step(kernel, opt_state)
            if verbose and i % 100 == 0:
                print(f"Iter {i}, loss = {loss:.4f}")

        # Mise à jour du kernel optimisé
        self.kernel = kernel

        # Calcul des quantités postérieures avec le kernel optimisé
        K = self.kernel(X)
        K = K + self.alpha * jnp.eye(len(X))
        L = jnp.linalg.cholesky(K)
        alpha_ = jax.scipy.linalg.cho_solve((L, True), y_norm)

        # Mettre à jour l'état
        self.X_train_ = X
        self.y_train_ = y_norm
        self.L_ = L
        self.alpha_ = alpha_
        self._y_mean = y_mean
        self._y_std = y_std

        return self

    @partial(jit, static_argnames=['return_std', 'return_cov'])
    def predict(self, X, return_std=False, return_cov=False):
     X = jnp.asarray(X)

     # Fonction pour calculer la diagonale (variance a priori) par point
     diag_fn = jax.vmap(lambda x: self.kernel(x, x))

     if self.X_train_ is None:
        # Modèle non entraîné : prior
        n_targets = 1 if self._y_mean is None else self._y_mean.shape[0]
        y_mean = jnp.zeros((X.shape[0], n_targets)).squeeze()
        if return_cov:
            y_cov = self.kernel(X)
            return y_mean, y_cov
        elif return_std:
            y_var = diag_fn(X)   # <-- remplace self.kernel.diag(X)
            return y_mean, jnp.sqrt(y_var)
        else:
            return y_mean
     else:
        # Modèle entraîné
        K_trans = self.kernel(X, self.X_train_)
        y_mean = K_trans @ self.alpha_
        y_mean = self._y_std * y_mean + self._y_mean
        if y_mean.ndim > 1 and y_mean.shape[1] == 1:
            y_mean = y_mean.squeeze(1)

        if return_cov:
            v = jax.scipy.linalg.solve_triangular(self.L_, K_trans.T, lower=True)
            y_cov = self.kernel(X) - v.T @ v
            y_cov = jnp.outer(y_cov, self._y_std**2).reshape(*y_cov.shape, -1)
            if y_cov.shape[-1] == 1:
                y_cov = y_cov.squeeze(-1)
            return y_mean, y_cov
        elif return_std:
            v = jax.scipy.linalg.solve_triangular(self.L_, K_trans.T, lower=True)
            # Utilisation de diag_fn pour la diagonale
            y_var = diag_fn(X) - jnp.einsum('ij,ji->i', v.T, v)
            y_var = y_var * self._y_std**2
            return y_mean, jnp.sqrt(y_var)
        else:
            return y_mean