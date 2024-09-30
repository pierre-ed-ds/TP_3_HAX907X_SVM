# TP No 3 : Support Vector Machine (SVM)

Ce projet utilise les **Support Vector Machines (SVM)** pour résoudre des problèmes de classification en Python avec la bibliothèque `scikit-learn`. L'objectif est d'explorer différentes approches d'apprentissage supervisé, de comprendre l'impact de plusieurs paramètres (tels que le noyau, la régularisation), ainsi que l'effet de la réduction de dimension sur la performance des modèles.

## Installation


Pour compiler le fichier `.tex` présent dans ce dépôt, veuillez suivre les étapes ci-dessous :


Assurez-vous d'avoir une distribution LaTeX installée sur votre machine. Voici quelques options :

- **Windows** : [MiKTeX](https://miktex.org/download) ou [TeX Live](https://www.tug.org/texlive/)
- **macOS** : [MacTeX](http://www.tug.org/mactex/)
- **Linux** : `TeX Live` (généralement disponible via votre gestionnaire de paquets)

**Clonez le dépôt:**
```
git clone git@github.com:pierre-ed-ds/TP_3_HAX907X_SVM.git
```

Installez les packages nécessaires puis compilez en faisant attention à l'arborescance des images.

Pour compiler le code de ```script.py```, veillez à installer les packages ```python``` suivants la ligne de commande:

```
pip install -r requirements.txt
```

## Contenu

Le projet se concentre sur les points suivants :

- **Classification avec SVM** : Utilisation de SVM pour classifier des jeux de données connus, comme le dataset Iris et un dataset de visages.
- **Noyaux SVM** : Comparaison entre différents types de noyaux (linéaire, polynomial) pour observer leur impact sur la précision du modèle.
- **Paramètres de régularisation** : Étude de l'influence du paramètre `C` sur la performance du modèle.
- **Réduction de dimension** : Amélioration des performances du modèle en réduisant la dimension des données avec l'algorithme **PCA** (Analyse en Composantes Principales).

## Technologies

- **Langage** : `Python`
- **Bibliothèque principale** : [scikit-learn](https://scikit-learn.org/stable/)
- **Autres dépendances** : NumPy, Matplotlib

## Autheur

[Pierre Dias](https://github.com/pierre-ed-ds)


