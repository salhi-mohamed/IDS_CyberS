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
    
    reader = pd.read_csv(f, chunksize=chunk_size, low_memory=False)
    processed_chunks = []

    for chunk in reader:
        # Nettoyer les noms de colonnes
        chunk.columns = chunk.columns.str.strip()
        
        # Détecter automatiquement la colonne Label
        label_col = [col for col in chunk.columns if "Label" in col][0]
        
        # Nettoyage : Inf → NaN, NaN → 0
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.fillna(0, inplace=True)

        # ⚠️ IMPORTANT : ne JAMAIS supprimer la colonne Label
        safe_drop_cols = []

        # Colonnes constantes
        constant_cols = [col for col in chunk.columns if chunk[col].nunique() <= 1]
        for col in constant_cols:
            if col != label_col:
                safe_drop_cols.append(col)

        # Colonnes vides
        empty_cols = [col for col in chunk.columns if chunk[col].isna().all()]
        for col in empty_cols:
            if col != label_col:
                safe_drop_cols.append(col)

        # Suppression sécurisée
        chunk.drop(columns=safe_drop_cols, inplace=True, errors='ignore')
        
        # ⚠️ NE PAS supprimer de lignes → enlève uniquement les doublons parfaits
        chunk.drop_duplicates(inplace=True)

        # Conversion object → numérique (hors Label)
        for col in chunk.select_dtypes(include=['object']).columns:
            if col != label_col:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0)
        
        # Encodage des colonnes catégorielles restantes
        categorical_cols = chunk.select_dtypes(include=['object']).columns.tolist()
        if label_col in categorical_cols:
            categorical_cols.remove(label_col)
        if categorical_cols:
            chunk = pd.get_dummies(chunk, columns=categorical_cols)
        
        # Scaling des colonnes numériques (hors Label)
        numeric_cols = chunk.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if label_col in numeric_cols:
            numeric_cols.remove(label_col)
        if numeric_cols:
            try:
                chunk[numeric_cols] = scaler.fit_transform(chunk[numeric_cols])
            except ValueError:
                # Cas colonnes vides ou lignes rares
                chunk[numeric_cols] = 0
        
        processed_chunks.append(chunk)
    
    # Concaténer les chunks traités
    df_cleaned = pd.concat(processed_chunks, ignore_index=True)
    
    # Conversion finale des objets → num
    for col in df_cleaned.select_dtypes(include=['object']).columns:
        if col != label_col:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0)
    
    df_cleaned.fillna(0, inplace=True)
    
    # Sauvegarde
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_cleaned.csv")
    df_cleaned.to_csv(output_path, index=False)
    print(f"✔ Fichier nettoyé et annoté sauvegardé : {output_path}")

print("\n✅ Phase 2 : Préparation du dataset terminée pour tous les fichiers.")
