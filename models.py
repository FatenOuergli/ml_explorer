import numpy as np
from sklearn.linear_model import LogisticRegression #regression logicstique
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.tree import DecisionTreeClassifier #Arbre de décision
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.svm import SVC #Support Vector Machine
from sklearn.naive_bayes import GaussianNB #Naive Bayes
 #ceux sont les sores obligatoire 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ModelTrainer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_available_models(self):
        return [
            "Régression Logistique",
            "K-Nearest Neighbors (KNN)",
            "Arbre de Décision",
            "Random Forest",
            "Support Vector Machine (SVM)",
            "Naive Bayes"
        ]

    def train_and_evaluate(self, model_name: str, **kwargs):
        #Initialis
        if model_name == "Régression Logistique":
            max_iter = kwargs.get('max_iter', 1000)
            model = LogisticRegression(max_iter=max_iter, random_state=42)
            
        elif model_name == "K-Nearest Neighbors (KNN)":
            n_neighbors = kwargs.get('n_neighbors', 5)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            
        elif model_name == "Arbre de Décision":
            max_depth = kwargs.get('max_depth', None) ##None = profondeur illimitée
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            
        elif model_name == "Random Forest":
            n_estimators = kwargs.get('n_estimators', 100)
            max_depth = kwargs.get('max_depth', None)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            
        elif model_name == "Support Vector Machine (SVM)":
            #kernel='rbf' est un bon défaut, C contrôle la régularisation
            kernel = kwargs.get('kernel', 'rbf')
            C = kwargs.get('C', 1.0)
            model = SVC(kernel=kernel, C=C, random_state=42, probability=True) # probability=True pour la courbe ROC
            
        elif model_name == "Naive Bayes":
            model = GaussianNB()
            
        else:
            raise ValueError(f"Modèle '{model_name}' non reconnu.")

        # 2. Entraînement du modèle
        model.fit(self.X_train, self.y_train)

        # 3. Prédictions sur l'ensemble de test
        y_pred = model.predict(self.X_test)

        #calcul des métriques d'évaluation 
        # On utilise average='weighted' pour gérer correctement les datasets multiclasse
        metrics = {
            "Accuracy": round(accuracy_score(self.y_test, y_pred), 4),
            "Precision": round(precision_score(self.y_test, y_pred, average='weighted', zero_division=0), 4),
            "Recall": round(recall_score(self.y_test, y_pred, average='weighted', zero_division=0), 4),
            "F1-Score": round(f1_score(self.y_test, y_pred, average='weighted', zero_division=0), 4)
        }

        #matrice de confusion 
        cm = confusion_matrix(self.y_test, y_pred)

        return {
            "model_name": model_name,
            "model_object": model,
            "predictions": y_pred,
            "metrics": metrics,
            "confusion_matrix": cm,
            "y_test": self.y_test #conservé pour générer la courbe ROC plus tard
        }

    def compare_models(self, model_list, **kwargs):
        results = []
        for model_name in model_list:
            # On extrait les paramètres spécifiques à ce modèle s'ils existent dans kwargs
            # Sinon, on passe un dictionnaire vide pour utiliser les défauts
            model_kwargs = kwargs.get(model_name, {})
            result = self.train_and_evaluate(model_name, **model_kwargs)
            results.append(result)
        return results