import sys
import os
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QComboBox, QLabel, QFileDialog, QTabWidget, 
    QTextEdit, QTableWidget, QTableWidgetItem, QGroupBox, 
    QMessageBox, QHeaderView, QListWidget, QAbstractItemView,
    QCheckBox, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt

# Imports Matplotlib pour l'intégration dans PyQt6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Imports de nos modules
from preprocessing import DataProcessor
from models import ModelTrainer
import visualization as viz

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = DataProcessor()
        self.trainer = None
        self.results = []
        
        self.setWindowTitle("ML Explorer - Tableau de Bord")
        self.resize(1100, 700)
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout = QVBoxLayout(central_widget)
        main_layout.addWidget(splitter)


        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(15)

        group_data = QGroupBox("1. Données")
        layout_data = QVBoxLayout(group_data)
        
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Dataset Scikit-learn", "Fichier CSV Local"])
        self.source_combo.currentTextChanged.connect(self.toggle_source_input)
        
        self.sklearn_combo = QComboBox()
        self.sklearn_combo.addItems(["iris", "wine", "breast_cancer", "digits"])
        
        self.csv_path_label = QLabel("Aucun fichier sélectionné")
        self.btn_browse = QPushButton("Parcourir...")
        self.btn_browse.clicked.connect(self.browse_csv)
        self.btn_browse.setVisible(False)
        
        self.btn_load = QPushButton("Charger les Données")
        self.btn_load.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 6px;")
        self.btn_load.clicked.connect(self.load_data)

        layout_data.addWidget(QLabel("Source :"))
        layout_data.addWidget(self.source_combo)
        layout_data.addWidget(self.sklearn_combo)
        layout_data.addWidget(self.csv_path_label)
        layout_data.addWidget(self.btn_browse)
        layout_data.addWidget(self.btn_load)
        left_layout.addWidget(group_data)

        group_prep = QGroupBox("2. Prétraitement")
        layout_prep = QVBoxLayout(group_prep)
        self.chk_scale = QCheckBox("Normaliser les données (StandardScaler)")
        self.chk_scale.setChecked(True)
        
        layout_split = QHBoxLayout()
        layout_split.addWidget(QLabel("Taille du Test Set:"))
        self.spin_test_size = QDoubleSpinBox()
        self.spin_test_size.setRange(0.1, 0.5)
        self.spin_test_size.setSingleStep(0.05)
        self.spin_test_size.setValue(0.2)
        layout_split.addWidget(self.spin_test_size)
        
        self.btn_preprocess = QPushButton("Appliquer le Prétraitement")
        self.btn_preprocess.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 6px;")
        self.btn_preprocess.clicked.connect(self.run_preprocessing)

        layout_prep.addWidget(self.chk_scale)
        layout_prep.addLayout(layout_split)
        layout_prep.addWidget(self.btn_preprocess)
        left_layout.addWidget(group_prep)

        group_models = QGroupBox("3. Algorithmes")
        layout_models = QVBoxLayout(group_models)
        
        self.list_models = QListWidget()
        self.list_models.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        available_models = ModelTrainer(None, None, None, None).get_available_models()
        self.list_models.addItems(available_models)
        
        self.spin_knn_k = QSpinBox()
        self.spin_knn_k.setRange(1, 50)
        self.spin_knn_k.setValue(5)
        self.spin_rf_est = QSpinBox()
        self.spin_rf_est.setRange(10, 500)
        self.spin_rf_est.setValue(100)
        self.spin_rf_est.setSingleStep(10)

        layout_models.addWidget(QLabel("Sélectionnez un ou plusieurs modèles :"))
        layout_models.addWidget(self.list_models)
        layout_models.addWidget(QLabel("Paramètres spécifiques :"))
        layout_models.addWidget(QLabel("  K pour KNN:"))
        layout_models.addWidget(self.spin_knn_k)
        layout_models.addWidget(QLabel("  Estimators pour Random Forest:"))
        layout_models.addWidget(self.spin_rf_est)
        
        self.btn_train = QPushButton("Lancer l'Entraînement")
        self.btn_train.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 6px;")
        self.btn_train.clicked.connect(self.run_training)
        
        self.btn_knn_curve = QPushButton("Courbe Accuracy vs K (KNN)")
        self.btn_knn_curve.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; padding: 6px;")
        self.btn_knn_curve.clicked.connect(self.plot_knn_curve)

        layout_models.addWidget(self.btn_train)
        layout_models.addWidget(self.btn_knn_curve)
        left_layout.addWidget(group_models)
        left_layout.addStretch()

        splitter.addWidget(left_panel)

    
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.tabs = QTabWidget()
        
        #donness
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.tabs.addTab(self.info_text, "📊 Infos Données")

        #mterique
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.tabs.addTab(self.metrics_text, "🏆 Métriques")

        #visualisation
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs.addTab(self.plot_widget, "📈 Visualisations")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_panel)
        
        # Ratio du split
        splitter.setSizes([300, 800])

    def toggle_source_input(self, source):
        is_sklearn = (source == "Dataset Scikit-learn")
        self.sklearn_combo.setVisible(is_sklearn)
        self.btn_browse.setVisible(not is_sklearn)
        self.csv_path_label.setVisible(not is_sklearn)

    def browse_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Sélectionner un CSV", "", "CSV Files (*.csv)")
        if file_path:
            self.csv_path_label.setText(os.path.basename(file_path))
            self._selected_csv_path = file_path

    def load_data(self):
        try:
            # 1. Charger les données
            if self.source_combo.currentText() == "Dataset Scikit-learn":
                self.processor.load_sklearn_dataset(self.sklearn_combo.currentText())
            else:
                if not hasattr(self, '_selected_csv_path'):
                    raise FileNotFoundError("Veuillez d'abord sélectionner un fichier CSV via le bouton 'Parcourir'.")
                
                import pandas as pd
                from PyQt6.QtWidgets import QInputDialog 
                
                temp_df = pd.read_csv(self._selected_csv_path)
                columns = temp_df.columns.tolist()
                
                # Demander la colonne cible
                target_col, ok = QInputDialog.getItem(
                    self, "Choix de la colonne cible", 
                    "Quelle est la colonne à prédire (Target) ?", 
                    columns, 0, False
                )
                
                if ok and target_col:
                    self.processor.load_csv(self._selected_csv_path, target_col)
                else:
                    return 

            info = self.processor.get_data_info()
            
            if "error" in info:
                raise ValueError(info["error"])

            shape = info['shape']
            missing = info['missing_values']
            
            total_missing = sum(missing.values())
            missing_str = f"{total_missing} valeur(s) manquante(s) au total.\n"
            for col, count in missing.items():
                if count > 0:
                    missing_str += f"   • {col} : {count}\n"
            if total_missing == 0:
                missing_str = "Aucune valeur manquante détectée. ✨\n"

            types_str = "\n".join([f"  • {k} : {v}" for k, v in info['dtypes'].items()])

            full_text = (
                f" Chargement réussi !\n\n"
                f" Dimensions : {shape[0]} lignes × {shape[1]} colonnes\n"
                f" Colonne cible : {self.processor.target_col}\n\n"
                f" Valeurs manquantes :\n{missing_str}\n"
                f" Types des variables :\n{types_str}"
            )

            self.info_text.setPlainText(full_text)
            
            self.tabs.setCurrentIndex(0) 
            #terminal!!
            print("DEBUG: Infos chargées et affichées avec succès.")

        except Exception as e:
            import traceback
            traceback.print_exc() 
            QMessageBox.critical(self, "Erreur de Chargement", f"Impossible de charger les données :\n\n{str(e)}")

    def run_preprocessing(self):
        if self.processor.df is None:
            QMessageBox.warning(self, "Attention", "Veuillez d'abord charger des données.")
            return
        try:
            self.processor.preprocess(
                test_size=self.spin_test_size.value(),
                scale=self.chk_scale.isChecked()
            )
            self.trainer = ModelTrainer(
                self.processor.X_train, self.processor.y_train,
                self.processor.X_test, self.processor.y_test
            )
            QMessageBox.information(self, "Succès", "Prétraitement terminé ! Vous pouvez lancer l'entraînement.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    def run_training(self):
        if self.trainer is None:
            QMessageBox.warning(self, "Attention", "Veuillez d'abord prétraiter les données.")
            return
        
        selected_models = [item.text() for item in self.list_models.selectedItems()]
        if not selected_models:
            QMessageBox.warning(self, "Attention", "Veuillez sélectionner au moins un modèle.")
            return

        try:
            self.results = []
            metrics_str = " RÉSULTATS DE L'ENTRAÎNEMENT\n" + "="*40 + "\n\n"
            
            params = {
                "K-Nearest Neighbors (KNN)": {"n_neighbors": self.spin_knn_k.value()},
                "Random Forest": {"n_estimators": self.spin_rf_est.value()}
            }

            for model_name in selected_models:
                res = self.trainer.train_and_evaluate(model_name, **params.get(model_name, {}))
                self.results.append(res)
                
                metrics_str += f"▶ {model_name}\n"
                for metric, val in res['metrics'].items():
                    metrics_str += f"   {metric}: {val:.4f}\n"
                metrics_str += "-" * 40 + "\n"

            self.metrics_text.setPlainText(metrics_str)
            
            if self.results:
                self.display_plot(viz.plot_confusion_matrix(
                    self.results[-1]['confusion_matrix'],
                    class_names=[str(c) for c in np.unique(self.processor.y)]
                ))
                
            QMessageBox.information(self, "Succès", "Entraînement terminé ! Consultez les onglets Métriques et Visualisations.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    def plot_knn_curve(self):
        if self.trainer is None:
            QMessageBox.warning(self, "Attention", "Veuillez d'abord prétraiter les données.")
            return
        try:
            fig = viz.plot_knn_accuracy_curve(
                self.processor.X_train, self.processor.y_train,
                self.processor.X_test, self.processor.y_test, max_k=20
            )
            self.display_plot(fig)
            self.tabs.setCurrentIndex(2) 
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    def display_plot(self, fig: Figure):
        for i in reversed(range(self.plot_layout.count())):
            widget = self.plot_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        canvas = FigureCanvas(fig)
        self.plot_layout.addWidget(canvas)