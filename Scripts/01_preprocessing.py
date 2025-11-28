import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Répertoires
data_dir = r"C:\Users\Lenovo\Desktop\Dataset"
output_dir = r"C:\Users\Lenovo\Desktop\Dataset_cleaned"
os.makedirs(output_dir, exist_ok=True)

# Paramètres
chunk_size = 500000  # ajuster selon RAM
scaler = StandardScaler()

# Lister tous les fichiers CSV
all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

for f in all_files:
    filename = os.path.basename(f)
    print(f"\n=== Traitement du fichier : {filename} ===")
    
    reader = pd.read_csv(f, chunksize=chunk_size)
    processed_chunks = []

    for chunk in reader:
        # Nettoyer les noms de colonnes
        chunk.columns = chunk.columns.str.strip()
        
        # Détecter automatiquement la colonne Label
        label_col = [col for col in chunk.columns if "Label" in col][0]
        
        # Remplacer Inf par NaN et remplir NaN par 0
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.fillna(0, inplace=True)
        
        # Supprimer colonnes constantes
        constant_cols = [col for col in chunk.columns if chunk[col].nunique() <= 1]
        chunk.drop(columns=constant_cols, inplace=True, errors='ignore')
        
        # Supprimer colonnes entièrement vides
        empty_cols = [col for col in chunk.columns if chunk[col].isna().all()]
        chunk.drop(columns=empty_cols, inplace=True, errors='ignore')
        
        # Supprimer doublons
        chunk.drop_duplicates(inplace=True)
        
        # Convertir colonnes object en numériques si possible
        for col in chunk.select_dtypes(include=['object']).columns:
            if col != label_col:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0)
        
        # Encodage des colonnes catégorielles restantes (optionnel)
        categorical_cols = chunk.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            chunk = pd.get_dummies(chunk, columns=categorical_cols)
        
        # Scaling des colonnes numériques (exclure Label)
        numeric_cols = chunk.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if label_col in numeric_cols:
            numeric_cols.remove(label_col)
        if numeric_cols:
            chunk[numeric_cols] = scaler.fit_transform(chunk[numeric_cols])
        
        processed_chunks.append(chunk)
    
    # Concaténer les chunks traités
    df_cleaned = pd.concat(processed_chunks, ignore_index=True)
    
    # Sauvegarder le fichier nettoyé
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_cleaned.csv")
    df_cleaned.to_csv(output_path, index=False)
    print(f"✔ Fichier nettoyé sauvegardé : {output_path}")

print("\n✅ Preprocessing batch terminé pour tous les fichiers.")
