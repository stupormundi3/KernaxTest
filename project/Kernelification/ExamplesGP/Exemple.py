
if __name__ == "__main__":
 from .Kernax import RBFKernel, Matern32Kernel  
 import numpy as np
 import matplotlib.pyplot as plt
 from .GP import GP
 #Generation of the data
 X = np.linspace(0, 10, 1000).reshape(-1, 1)
 y = np.squeeze(X * np.sin(X))
 rng = np.random.RandomState(1)
 training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
 X_train, y_train = X[training_indices], y[training_indices]


 kernel = Matern32Kernel(length_scale=1.0)  

 gaussian_process = GP(Kernel=kernel, Alpha=1e-5, NormalizeObs=True)

#Training
 gaussian_process.fit(X_train, y_train, num_iters=500, learning_rate=0.01, progress=True)


 mean_prediction, std_prediction = gaussian_process.predict(X, Return_std=True)


 plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
 plt.scatter(X_train, y_train, label="Observations")
 plt.plot(X, mean_prediction, label="Mean prediction")
 plt.fill_between(
     X.ravel(),
     mean_prediction - 1.96 * std_prediction,
     mean_prediction + 1.96 * std_prediction,
     alpha=0.5,
     label=r"95% confidence interval",
 )
 plt.legend()
 plt.xlabel("$x$")
 plt.ylabel("$f(x)$")
 plt.title("Gaussian process regression with JAX")
 plt.show()