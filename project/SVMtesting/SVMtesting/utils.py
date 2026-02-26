# -*- coding: utf-8 -*-
"""
Module utilitaire pour le chargement et la préparation des données.
"""

import numpy as np

def generer_donnees_demo(n_samples=100, bruit=0.1):
    """
    Génère un jeu de données synthétique pour démontrer l'utilisation du SVM.

    Il s'agit d'un problème de classification binaire simple avec deux
    caractéristiques, basé sur l'exemple des "lunes" de scikit-learn.

    :param n_samples: Le nombre total d'échantillons à générer.
    :type n_samples: int
    :param bruit: L'écart-type du bruit gaussien à ajouter aux données.
    :type bruit: float

    :returns: Un tuple (X, y) où X est la matrice des caractéristiques (n_samples, 2)
              et y est le vecteur des étiquettes (n_samples,).
    :rtype: tuple
    """
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n_samples, noise=bruit, random_state=42)
    return X, y