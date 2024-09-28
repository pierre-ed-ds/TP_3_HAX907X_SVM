#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from svm_source import *
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from time import time

scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')

#%%
###############################################################################
#               Iris Dataset
###############################################################################

iris = datasets.load_iris()
X = iris.data
X = scaler.fit_transform(X)
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]

# split train test
iris = datasets.load_iris()
X = iris.data
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]


#%%
#Q1 
#On boucle 1000 fois pour avoir une moyenne des scores
# Initialisation pour stocker les scores
train_scores = []
test_scores = []

# Répéter 1000 fois
for i in range(1000):
    # Shuffle et split des données
    X, y = shuffle(X, y, random_state=i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)

    # Initialisation du modèle SVM avec noyau linéaire
    clf_linear = SVC(kernel='linear', C=1.0)

    # Entraînement du modèle
    clf_linear.fit(X_train, y_train)

    # Calcul et stockage des scores
    train_scores.append(clf_linear.score(X_train, y_train))
    test_scores.append(clf_linear.score(X_test, y_test))

# Calcul des moyennes
mean_train_score = np.mean(train_scores)
mean_test_score = np.mean(test_scores)

# Affichage des moyennes
print(f'Moyenne des scores d\'entraînement pour le noyau linéaire: {mean_train_score}')
print(f'Moyenne des scores de test pour le noyau linéaire: {mean_test_score}')

#%%
# Q2 Noyau Polynomial
Cs = list(map(float, np.logspace(-3, 3, 5)))
gammas = list(map(float, 10. ** np.arange(1, 2)))
degrees = list(map(int, np.r_[1, 2, 3]))

# Définition de la grille de paramètres
parameters = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}

# Utilisation de GridSearchCV
clf_poly = GridSearchCV(SVC(), param_grid=parameters, n_jobs=-1)
clf_poly.fit(X_train, y_train)

# Affichage des meilleurs paramètres avec les bons types
print(clf_poly.best_params_)


#%%
# display your results using frontiere

def f_linear(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_linear.predict(xx.reshape(1, -1))

def f_poly(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_poly.predict(xx.reshape(1, -1))

plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(131)
plot_2d(X, y)
plt.title("iris dataset")

plt.subplot(132)
frontiere(f_linear, X, y)
plt.title("linear kernel")

plt.subplot(133)
frontiere(f_poly, X, y)

plt.title("polynomial kernel")
plt.tight_layout()
plt.draw()

#%%

###############################################################################
#               Face Recognition Task
###############################################################################
"""
The dataset used in this example is a preprocessed excerpt
of the "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  _LFW: http://vis-www.cs.umass.edu/lfw/
"""

####################################################################
# Chargement des données avec plus de 70 images par personnes
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)
# data_home='.'

# introspect the images arrays to find the shapes (for plotting)
images = lfw_people.images
n_samples, h, w, n_colors = images.shape

# the label to predict is the id of the person
target_names = lfw_people.target_names.tolist()

####################################################################
# Choix des personnes a classifier
names = ['Tony Blair', 'Colin Powell']
idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(int)

# echantillon 
plot_gallery(images, np.arange(12))
plt.show()

#%%
####################################################################
# Extract features

# features using only illuminations
X = (np.mean(images, axis=3)).reshape(n_samples, -1)

# Normalisation
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

#%%
indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[
    train_idx, :, :, :], images[test_idx, :, :, :]

#%%
# Q4
print("--- Linear kernel ---")
print("Fitting the classifier to the training set")
t0 = time()

# tester tout les Cs dans la range
Cs = 10. ** np.arange(-5, 6)
erreurs= []
for C in Cs:
    # Initialiser et entraîner le modèle SVM avec un noyau linéaire
    clf = SVC(kernel='linear', C=C)
    clf.fit(X_train, y_train)

    # Évaluer la performance sur l'ensemble d'entraînement
    score = clf.score(X_train, y_train)
    erreur = 1 - score
    erreurs.append(erreur)

ind = np.argmin(erreurs)
print("Best C: {}".format(Cs[ind]))

plt.figure()
plt.plot(Cs, erreurs)
plt.xlabel("Parametres de regularisation C")
plt.ylabel("Erreur d'apprentissage")
plt.xscale("log")
plt.tight_layout()
plt.show()
print("Plus petite erreur: {}".format(np.min(erreur)))

print("Predicting the people names on the testing set")
print("temps: %0.3fs" % (time() - t0))

#%%
# prédiction des nom par rapport aux images
t0 = time()
clf = SVC(kernel='linear', C=Cs[ind])
clf.fit(X_train, y_train) #on fit pour avoir les coefficients

score = clf.score(X_test, y_test)

print("done in %0.3fs" % (time() - t0))
# The chance level is the accuracy that will be reached when constantly predicting the majority class.
print("Chance level : %s" % max(np.mean(y), 1. - np.mean(y)))
print("Accuracy : %s" % score)

#%%
####################################################################
# Qualitative evaluation of the predictions using matplotlib
y_pred = clf.predict(X_test)
prediction_titles = [title(y_pred[i], y_test[i], names)
                     for i in range(y_pred.shape[0])]

plot_gallery(images_test, prediction_titles)
plt.show()

####################################################################
# Look at the coefficients
plt.figure()
plt.imshow(np.reshape(clf.coef_, (h, w)))
plt.show()


#%%
# Q5
#Comparaison d'un modèle normale a un modèle avec du bruit
def run_svm_cv(_X, _y):
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]

    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    _svr = svm.SVC()
    _clf_linear = GridSearchCV(_svr, _parameters)
    _clf_linear.fit(_X_train, _y_train)

    print('Generalization score for linear kernel: %s, %s \n' %
          (_clf_linear.score(_X_train, _y_train), _clf_linear.score(_X_test, _y_test)))

print("Score sans variable de nuisance")
run_svm_cv(X, y) 

print("Score avec variable de nuisance")
n_features = X.shape[1]
# On rajoute des variables de nuisances
sigma = 1
noise = sigma * np.random.randn(n_samples, 300, )
X_noisy = np.concatenate((X, noise), axis=1)
X_noisy = X_noisy[np.random.permutation(X.shape[0])]
run_svm_cv(X_noisy, y) 

#%%
# Q6
# c'est extremement long, mon ordinateur plante, sj'ai du le faire sur google colab
print("Score apres reduction de dimension")

n_components = 50  # jouer avec ce parametre
pca = PCA(n_components=n_components,svd_solver="randomized").fit(X_noisy)
# On applique PCA sur les données avec bruit
X_noisy_pca = pca.fit_transform(X_noisy)
# On utilise les données transformées (réduites) dans run_svm_cv

run_svm_cv(X_noisy_pca, y)
# %%
#Q6 bis
n_components_list = np.linspace(90, 300, 10, dtype=int)  # 10 valeurs de n_components entre 80 et 380
train_scores = []
test_scores = []

# Fonction run_svm_cv réutilisée avec une petite modification pour capturer les scores
def run_svm_cv_with_scores(_X, _y):
    _indices = np.random.permutation(_X.shape[0])
    _train_idx, _test_idx = _indices[:_X.shape[0] // 2], _indices[_X.shape[0] // 2:]
    _X_train, _X_test = _X[_train_idx, :], _X[_test_idx, :]
    _y_train, _y_test = _y[_train_idx], _y[_test_idx]

    _parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    _svr = svm.SVC()
    _clf_linear = GridSearchCV(_svr, _parameters)
    _clf_linear.fit(_X_train, _y_train)

    train_score = _clf_linear.score(_X_train, _y_train)
    test_score = _clf_linear.score(_X_test, _y_test)
    
    return train_score, test_score

# Boucle sur différentes valeurs de n_components
for n_components in n_components_list:
    print(f"Testing with n_components = {n_components}")
    
    # Appliquer PCA pour réduire les données
    pca = PCA(n_components=n_components).fit(X_noisy)
    X_noisy_pca = pca.transform(X_noisy)
    
    # Appeler run_svm_cv_with_scores et récupérer les scores
    train_score, test_score = run_svm_cv_with_scores(X_noisy_pca, y)
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    print(f"Train score: {train_score}, Test score: {test_score}")

# Plot des scores
plt.figure(figsize=(10, 6))
plt.plot(n_components_list, train_scores, label='Train score', marker='o')
plt.plot(n_components_list, test_scores, label='Test score', marker='o')
plt.xlabel('Number of PCA components')
plt.ylabel('Score')
plt.title('Impact of PCA Components on SVM Performance')
plt.legend()
plt.grid(True)
plt.show()

