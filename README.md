# TP No 3 : Support Vector Machine (SVM)

Ce projet utilise les **Support Vector Machines (SVM)** pour résoudre des problèmes de classification en Python avec la bibliothèque `scikit-learn`. L'objectif est d'explorer différentes approches d'apprentissage supervisé, de comprendre l'impact de plusieurs paramètres (tels que le noyau, la régularisation et la pondération des classes), ainsi que l'effet de la réduction de dimension sur la performance des modèles.

## Contenu

Le projet se concentre sur les points suivants :

- **Classification avec SVM** : Utilisation de SVM pour classifier des jeux de données connus, comme le dataset Iris et un dataset de visages.
- **Noyaux SVM** : Comparaison entre différents types de noyaux (linéaire, polynomial) pour observer leur impact sur la précision du modèle.
- **Paramètres de régularisation** : Étude de l'influence du paramètre `C` sur la performance du modèle, en particulier dans des situations de données déséquilibrées.
- **Données déséquilibrées** : Exploration de la façon dont les SVM se comportent avec des classes de tailles très inégales et comment ajuster le modèle pour gérer ces déséquilibres (ex. : pondération des classes).
- **Réduction de dimension** : Amélioration des performances du modèle en réduisant la dimension des données avec l'algorithme **PCA** (Analyse en Composantes Principales).

## Technologies

- **Langage** : `Python`
- **Bibliothèque principale** : [scikit-learn](https://scikit-learn.org/stable/)
- **Autres dépendances** : NumPy, Matplotlib

## Objectifs

- **Compréhension des SVM** : Apprendre comment les SVM fonctionnent pour la classification et l'importance des différents noyaux et paramètres.
- **Impact de la régularisation** : Observer l'effet de la régularisation sur les performances du modèle dans des environnements équilibrés et déséquilibrés.
- **Réduction de dimension** : Utiliser des techniques de réduction de dimension pour améliorer les prédictions, particulièrement sur des jeux de données avec des caractéristiques inutiles ou du bruit.
