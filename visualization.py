import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

#matrice de confusion
def plot_confusion_matrix(cm, class_names):
    """
    Génère une Figure Matplotlib de la matrice de confusion avec Seaborn.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # seaborn (heatmap elegante)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                cbar_kws={'label': 'Nombre de prédictions'})
    
    ax.set_xlabel('Classe Prédite', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classe Réelle', fontsize=12, fontweight='bold')
    ax.set_title('Matrice de Confusion', fontsize=14, fontweight='bold', pad=15)
    
    fig.tight_layout()
    return fig

def plot_roc_curve(model, X_test, y_test):
    """
    Génère la courbe ROC. Gère automatiquement les problèmes binaires et multiclasse (One-vs-Rest).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    classes = np.unique(y_test)
    
    if not hasattr(model, "predict_proba"):
        ax.text(0.5, 0.5, "Ce modèle ne supporte pas les probabilités\npour la courbe ROC.", 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
        ax.set_title('Courbe ROC Non Disponible')
        return fig

    if len(classes) == 2:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkorange', lw=3, 
                label=f'Courbe ROC (AUC = {roc_auc:.3f})')
                
    else:
        y_test_bin = label_binarize(y_test, classes=classes)
        n_classes = len(classes)
        y_prob = model.predict_proba(X_test)
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
        for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC Classe {classes[i]} (AUC = {roc_auc[i]:.2f})')
            
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Hasard (AUC = 0.5)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
    ax.set_ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
    ax.set_title('Courbe Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    
    fig.tight_layout()
    return fig


def plot_scores_comparison(results_list):
    """
    Génère un graphique en barres groupées comparant les modèles sur les 4 métriques.
    results_list: Liste des dictionnaires retournés par ModelTrainer.compare_models()
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [res['model_name'] for res in results_list]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336'] 
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [res['metrics'][metric] for res in results_list]
        bars = ax.bar(x + i*width, values, width, label=metric, color=colors[i], edgecolor='black', linewidth=0.5)
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', 
                    ha='center', va='bottom', fontsize=9)
            
    ax.set_ylabel('Score (0 à 1)', fontsize=12)
    ax.set_title('Comparaison des Performances des Modèles', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    fig.tight_layout()
    return fig

#vari
def plot_feature_importance(model, feature_names):
    """
    Génère un graphique de l'importance des variables (pour Random Forest / Arbre de décision).
    """
    if not hasattr(model, "feature_importances_"):
        return None
        
    fig, ax = plt.subplots(figsize=(8, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    ax.bar(range(len(importances)), importances[indices], color='teal', edgecolor='black')
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_title('Importance des Variables (Feature Importance)', fontsize=14, fontweight='bold')
    ax.set_ylabel("Importance")
    
    fig.tight_layout()
    return fig

#knn addition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def plot_knn_accuracy_curve(X_train, y_train, X_test, y_test, max_k=20):
  
    fig, ax = plt.subplots(figsize=(8, 5))
    k_values = range(1, max_k + 1)
    accuracies = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
    ax.plot(k_values, accuracies, marker='o', color='purple', linewidth=2, markersize=6)
    ax.set_xlabel('Nombre de voisins (K)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Évolution de l\'Accuracy en fonction de K (KNN)', fontsize=14, fontweight='bold')
    ax.set_xticks(k_values)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    best_k = k_values[np.argmax(accuracies)]
    best_acc = max(accuracies)
    ax.axvline(best_k, color='red', linestyle='dashed', linewidth=1)
    ax.text(best_k, best_acc, f'  Meilleur K={best_k}\n  Accuracy={best_acc:.3f}', 
            verticalalignment='bottom', color='red', fontweight='bold')
    
    fig.tight_layout()
    return fig