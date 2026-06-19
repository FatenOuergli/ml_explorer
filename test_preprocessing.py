from preprocessing import DataProcessor

CSV_PATH = "votre_fichier_kaggle.csv" 

TARGET_COLUMN = "Admission" 

print(f"Chargement de {CSV_PATH}...")
processor = DataProcessor()
processor.load_csv(CSV_PATH, TARGET_COLUMN)

print("Chargé. Aperçu des types :")
print(processor.df.dtypes)

print("\nLancement du prétraitement...")
try:
    # On teste avec et sans normalisation
    result = processor.preprocess(test_size=0.2, scale=True) 
    print(" SUCCÈS !", result['message'])
    print(f"Forme de X_train : {processor.X_train.shape}")
except Exception as e:
    print(f"ERREUR CRITIQUE : {e} !")