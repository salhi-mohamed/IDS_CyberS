import os
import glob
import pandas as pd
import numpy as np

# Répertoires
cleaned_dir = r"C:\Users\Lenovo\Desktop\Dataset_cleaned"
output_dir = r"C:\Users\Lenovo\Desktop\Dataset_prepared"
os.makedirs(output_dir, exist_ok=True)

# Lister tous les fichiers nettoyés
all_files = sorted(glob.glob(os.path.join(cleaned_dir, "*_cleaned.csv")))
print(f"=== {len(all_files)} fichiers trouvés dans {cleaned_dir} ===")

dfs = []

for f in all_files:
    filename = os.path.basename(f)
    print(f"\n[INFO] Lecture du fichier : {filename}")
    df = pd.read_csv(f, low_memory=False)
    
    # Nettoyer les colonnes
    df.columns = df.columns.str.strip()
    
    # Détecter automatiquement la colonne Label
    label_cols = [col for col in df.columns if "Label" in col]
    if not label_cols:
        print(f"[WARNING] Aucun label trouvé dans {filename}, fichier ignoré.")
        continue
    label_col = label_cols[0]
    
    # Compter lignes avant dropna
    n_before = df.shape[0]
    
    # Supprimer les lignes où le label est manquant
    df = df.dropna(subset=[label_col])
    n_after = df.shape[0]
    
    print(f"[INFO] Lignes avant dropna: {n_before}, après dropna: {n_after}, supprimées: {n_before - n_after}")
    
    dfs.append(df)

# Concaténer tous les fichiers
df_global = pd.concat(dfs, ignore_index=True)
print(f"\n[INFO] Dataset global créé avec {df_global.shape[0]} lignes et {df_global.shape[1]} colonnes")

# Transformer les labels en "normal"/"anormal"
y_global = df_global[label_col].apply(lambda x: "normal" if x == "BENIGN" else "anormal")

# Supprimer la colonne Label de X
X_global = df_global.drop(columns=[label_col])

# ⚠️ Réinitialiser les index pour éviter tout décalage
X_global = X_global.reset_index(drop=True)
y_global = y_global.reset_index(drop=True)

# Vérification
print("\n[INFO] Vérification des tailles après reset_index")
print(f"X_global.shape = {X_global.shape}")
print(f"y_global.shape = {y_global.shape}")

if X_global.shape[0] != y_global.shape[0]:
    print("[ERROR] Taille de X et y NON concordante !")
else:
    print("[SUCCESS] Taille de X et y concordante ✔")

# Distribution des labels
print("\n[INFO] Distribution des labels y_global :")
print(y_global.value_counts())

# Sauvegarde
X_global_path = os.path.join(output_dir, "X_global.csv")
y_global_path = os.path.join(output_dir, "y_global.csv")
X_global.to_csv(X_global_path, index=False)
y_global.to_csv(y_global_path, index=False)

print(f"\n✔ X_global sauvegardé : {X_global_path}")
print(f"✔ y_global sauvegardé : {y_global_path}")
print("\n✅ Dataset global prêt pour ML")
