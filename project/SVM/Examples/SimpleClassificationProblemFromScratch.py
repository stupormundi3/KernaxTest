from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from SVM import SVM

# Creating dataset
# creating X and y values
nSamples = 100
spanLen = 2
X = np.zeros((nSamples, 2))
y = np.zeros((nSamples, ))
c = 0
# Generation of our dataset, each point can either have 1 or 0 as his class.
for i in range(nSamples):
 
  a, b = [np.random.uniform(-spanLen, spanLen) for _ in ['x', 'y']]

  X[i][0], X[i][1] = a, b
  y[i] = 1 if a*a + b*b < 1.2*1.2 else 0
  
print(type(X))
print(X)
print(type(y))
print(y)
# Classes 1 and -1
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=40)
svm = SVM()
svm.kernel = 'SE'
w, b, losses = svm.fit(X_train, y_train)
prediction = svm.predict(X_test)

# Loss value
lss = losses.pop()

print("Loss:", lss)
print("Prediction:", prediction)
print("Accuracy:", accuracy_score(prediction, y_test))
print("w, b:", [w, b])

# Visualizing the scatter plot of the dataset
def visualize_dataset():
    plt.scatter(X[:, 0], X[:, 1], c=y)


# Visualizing SVM
def visualize_poly_svm(svm, X_train, y_train, X_test, y_test):
    """Visualise la frontière de décision du SVM avec tous les points d'entraînement et de test"""
    
    # Créer la figure
    plt.figure(figsize=(10, 8))
    
    # limites du graphe
    all_features_1 = np.concatenate([X_train[:, 0], X_test[:, 0]])
    all_features_2 = np.concatenate([X_train[:, 1], X_test[:, 1]])
    
    x_min = all_features_1.min() - 0.5
    x_max = all_features_1.max() + 0.5
    y_min = all_features_2.min() - 0.5
    y_max = all_features_2.max() + 0.5
    

    x_points = np.linspace(x_min, x_max, 300)
    y_points = np.linspace(y_min, y_max, 300)
    grid_x, grid_y = np.meshgrid(x_points, y_points)
    

    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    # prediction for each point
    grid_predictions = svm.predict(grid_points)
    grid_predictions = grid_predictions.reshape(grid_x.shape)
    
    
    plt.contourf(grid_x, grid_y, grid_predictions, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.contour(grid_x, grid_y, grid_predictions, colors='black', linewidths=1, alpha=0.7)
    
    # compute les points d'entrainement
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, 
               edgecolors='black', s=60, alpha=0.8, label='Données d\'entraînement')
    
    # les tests points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, 
               marker='s', edgecolors='black', s=50, alpha=0.8, label='Données de test')
    
    # Calculer la précision pour l'affichage
    test_predictions = svm.predict(X_test)
    accuracy = accuracy_score(test_predictions, y_test)
    
    # labels etc
    plt.xlabel("Première caractéristique")
    plt.ylabel("Deuxième caractéristique")
    plt.title(f"SVM avec noyau {"Square Exponential"}\nPrécision sur les tests: {accuracy:.3f}")
    plt.legend()
    plt.colorbar(label='Classe prédite')
    

    plt.tight_layout()
    plt.show()

visualize_dataset()
visualize_poly_svm(svm, X_train, y_train, X_test, y_test)

