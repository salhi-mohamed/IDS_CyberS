import os
import glob
import pandas as pd
import numpy as np

cleaned_dir = r"C:\Users\Lenovo\Desktop\Dataset_cleaned"

# Lister tous les fichiers nettoyés
files = sorted(glob.glob(os.path.join(cleaned_dir, "*.csv")))

for f in files:
    filename = os.path.basename(f)
    print(f"\n=== Vérification du fichier : {filename} ===")
    
    df = pd.read_csv(f)
    
    # Vérifier NaN et Inf
    nan_count = df.isna().sum().sum()
    inf_count = ((df == np.inf) | (df == -np.inf)).sum().sum()
    
    if nan_count == 0 and inf_count == 0:
        print("✔ Pas de NaN ou Inf")
    else:
        print(f"⚠ Attention : {nan_count} NaN, {inf_count} Inf")
    
    # Colonnes constantes
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        print(f"⚠ Colonnes constantes détectées : {constant_cols}")
    else:
        print("✔ Pas de colonnes constantes")
    
    # Colonnes vides
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    if empty_cols:
        print(f"⚠ Colonnes entièrement vides : {empty_cols}")
    else:
        print("✔ Pas de colonnes vides")
    
    # Types de colonnes
    print("Types de colonnes :")
    print(df.dtypes.value_counts())
    
    # Dimensions
    print(f"Shape : {df.shape}")
    print(f"Exemple de valeurs uniques pour quelques colonnes :")
    for col in df.columns[:5]:  # montrer seulement les 5 premières colonnes
        print(f"{col}: {df[col].nunique()} valeurs uniques")
