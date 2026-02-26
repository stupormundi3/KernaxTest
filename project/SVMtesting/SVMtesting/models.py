# -*- coding: utf-8 -*-
"""
Ce module contient les implémentations de classifieurs SVM.
"""

import numpy as np
from sklearn import svm
from sklearn.base import BaseEstimator, ClassifierMixin

class MonSuperSVM(BaseEstimator, ClassifierMixin):
    """
    Une classe wrapper pour un classifieur SVM avec un noyau personnalisable.

    Cette classe utilise scikit-learn en backend mais ajoute des fonctionnalités
    de logging et de validation spécifiques au projet.

    :param kernel: Le type de noyau à utiliser ('linear', 'rbf', 'poly').
    :type kernel: str, optional
    :param C: Paramètre de régularisation. Doit être strictement positif.
    :type C: float, optional
    :param gamma: Coefficient pour les noyaux 'rbf', 'poly' et 'sigmoid'.
    :type gamma: {'scale', 'auto'} or float, optional

    :raises ValueError: Si la valeur de `C` est négative ou nulle.

    :example:
    >>> modele = MonSuperSVM(kernel='rbf', C=1.0, gamma='auto')
    >>> modele.fit(X_train, y_train)
    >>> predictions = modele.predict(X_test)
    """

    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self._model = None

    def fit(self, X, y):
        """
        Entraîne le modèle SVM sur les données fournies.

        Cette méthode appelle l'implémentation de scikit-learn après avoir
        effectué quelques vérifications de base.

        :param X: Les caractéristiques d'entraînement, de forme (n_échantillons, n_caractéristiques).
        :type X: np.ndarray
        :param y: Les étiquettes cibles, de forme (n_échantillons,).
        :type y: np.ndarray

        :returns: L'instance elle-même (pour le chaînage).
        :rtype: MonSuperSVM
        """
        if self.C <= 0:
            raise ValueError("Le paramètre C doit être strictement positif.")

        print(f"Entraînement du SVM avec noyau {self.kernel}...")
        self._model = svm.SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        self._model.fit(X, y)
        print("Entraînement terminé.")
        return self

    def predict(self, X):
        """
        Prédit les étiquettes pour de nouveaux échantillons.

        :param X: Les données à prédire, de forme (n_échantillons, n_caractéristiques).
        :type X: np.ndarray

        :returns: Les étiquettes prédites.
        :rtype: np.ndarray

        :raises sklearn.exceptions.NotFittedError: Si le modèle n'a pas été entraîné au préalable.
        """
        if self._model is None:
            raise AttributeError("Vous devez appeler la méthode 'fit' avant de faire des prédictions.")
        return self._model.predict(X)