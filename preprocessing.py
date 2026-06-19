import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataProcessor:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_col = None

    def load_sklearn_dataset(self, dataset_name: str):
        datasets = {
            "iris": load_iris, "wine": load_wine,
            "breast_cancer": load_breast_cancer, "digits": load_digits
        }
        if dataset_name not in datasets:
            raise ValueError(f"Dataset inconnu.")
        
        data = datasets[dataset_name]()
        self.df = pd.DataFrame(data.data, columns=data.feature_names)
        self.target_col = "target"
        self.df[self.target_col] = data.target
        return self.df

    def load_csv(self, filepath: str, target_col: str):
        self.df = pd.read_csv(filepath)
        self.target_col = target_col
        
        if self.target_col not in self.df.columns:
            raise ValueError(f"La colonne '{target_col}' n'existe pas.")
        return self.df

    def get_data_info(self):
        if self.df is None: return {"error": "Aucune donnée."}
        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "preview": self.df.head(5).to_dict(orient="records"),
            "missing_values": self.df.isnull().sum().to_dict()
        }

    def preprocess(self, test_size=0.2, random_state=42, handle_missing="mean", scale=True):
        X = self.df.drop(columns=[self.target_col]).copy()
        y = self.df[self.target_col].copy()

        #supprime les colonnes uniques qui faussent le ML
        for col in X.columns:
            if X[col].nunique() == len(X) and X[col].dtype in ['int64', 'object']:
                print(f"⚠️ [Auto-Nettoyage] Colonne '{col}' détectée comme un ID. Supprimée.")
                X = X.drop(columns=[col])

        # gestion des valeurs manquantes (Méthode Pandas native, plus stable)
        for col in X.columns:
            if X[col].dtype in ['object', 'category', 'bool']:
                mode_val = X[col].mode()[0] if not X[col].mode().empty else "Unknown"
                X[col] = X[col].fillna(mode_val)
            else:
                if handle_missing == "mean":
                    X[col] = X[col].fillna(X[col].mean())
                elif handle_missing == "median":
                    X[col] = X[col].fillna(X[col].median())

        #encodage (NaN résiduels)
        for col in X.columns:
            if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        #encodage de la cible si c'est du texte
        if y.dtype == 'object' or str(y.dtype) == 'category':
            le_y = LabelEncoder()
            y = le_y.fit_transform(y.astype(str))

        #train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y if len(np.unique(y)) > 1 else None
        )

        #normalisation
        if scale:
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
        else:
            self.X_train = self.X_train.values
            self.X_test = self.X_test.values

        return {"message": "Prétraitement terminé avec succès !"}