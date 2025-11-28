import os
import glob
import pandas as pd

# Répertoires
cleaned_dir = r"C:\Users\Lenovo\Desktop\Dataset_cleaned"
output_dir = r"C:\Users\Lenovo\Desktop\Dataset_prepared"
os.makedirs(output_dir, exist_ok=True)

# Lister tous les fichiers nettoyés
all_files = sorted(glob.glob(os.path.join(cleaned_dir, "*_cleaned.csv")))

print("=== Concaténation des fichiers nettoyés ===")

# Liste pour stocker chaque dataframe
dfs = []

for f in all_files:
    filename = os.path.basename(f)
    print(f"Lecture du fichier : {filename}")
    df = pd.read_csv(f, low_memory=False)
    dfs.append(df)

# Concaténer tous les fichiers
df_global = pd.concat(dfs, ignore_index=True)
print(f"\n✔ Dataset global créé avec {df_global.shape[0]} lignes et {df_global.shape[1]} colonnes")

# Détecter la colonne Label
label_col = [col for col in df_global.columns if "Label" in col][0]

# Transformer les labels : BENIGN → normal, autres → anormal
y_global = df_global[label_col].apply(lambda x: "normal" if x == "BENIGN" else "anormal")

# Supprimer la colonne Label de X
X_global = df_global.drop(columns=[label_col])

# Vérification du nombre de lignes
print("\n=== Vérification du dataset global ===")
if X_global.shape[0] == y_global.shape[0]:
    print(f"✔ Nombre de lignes correct : {X_global.shape[0]} lignes pour X et y")
else:
    print(f"⚠ Nombre de lignes incohérent : X={X_global.shape[0]}, y={y_global.shape[0]}")

# Vérification de la distribution des labels
print("\nDistribution des labels y_global :")
print(y_global.value_counts())

# Sauvegarder X et y
X_global_path = os.path.join(output_dir, "X_global.csv")
y_global_path = os.path.join(output_dir, "y_global.csv")

X_global.to_csv(X_global_path, index=False)
y_global.to_csv(y_global_path, index=False)

print(f"\n✔ X_global sauvegardé : {X_global_path}")
print(f"✔ y_global sauvegardé : {y_global_path}")
print("\n✅ Concaténation et annotation terminées, dataset global prêt pour ML")
