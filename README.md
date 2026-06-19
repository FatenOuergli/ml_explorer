# ML Explorer

**Application Python de comparaison d'algorithmes de Machine Learning**


##  À propos du projet

**ML Explorer** est une application Python avec interface graphique qui permet à un utilisateur, même non-expert, de :

- Charger des jeux de données (CSV ou datasets Scikit-learn)
- Prétraiter les données (nettoyage, encodage, normalisation)
- Entraîner plusieurs algorithmes de classification
- Visualiser et comparer leurs performances

Ce projet a été réalisé dans le cadre du module de **Programmation Python** à EPI.

##  Fonctionnalités

### Gestion des données
- Chargement de fichiers CSV locaux
- Accès aux datasets intégrés Scikit-learn : **Iris, Wine, Breast Cancer, Digits**
- Aperçu des données (shape, types, valeurs manquantes)

### Prétraitement automatique
- Détection et suppression des colonnes "ID"
- Gestion des valeurs manquantes
- Encodage des variables catégorielles
- Normalisation optionnelle (StandardScaler)
- Séparation Train/Test avec stratification

### Algorithmes implémentés
- Régression Logistique
- K-Nearest Neighbors (KNN)
- Arbre de Décision
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes

### Évaluation & Visualisation
- Calcul des métriques : **Accuracy, Precision, Recall, F1-Score**
- Matrice de confusion (heatmap)
- Courbe ROC (binaire et multiclasse)
- Comparaison graphique des modèles
- **Bonus** : Courbe Accuracy = f(K) pour le KNN

### Interface graphique moderne
- Tableau de bord PyQt6 avec panneau de contrôle
- Infobulles explicatives pour les paramètres (pédagogie)
- Affichage dynamique des graphiques Matplotlib/Seaborn
## Installation

### 1. Cloner le dépôt

#git clone https://github.com/VOTRE_USERNAME/ML_Explorer.git
#cd ML_Explorer
#python main.py
