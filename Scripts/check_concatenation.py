import os
import pandas as pd

# Répertoire des fichiers préparés
prepared_dir = r"C:\Users\Lenovo\Desktop\Dataset_prepared"

# Chemins des fichiers
X_path = os.path.join(prepared_dir, "X_global.csv")
y_path = os.path.join(prepared_dir, "y_global.csv")

print("=== Vérification des fichiers concaténés ===")

# Vérifier que les fichiers existent
if not os.path.exists(X_path):
    print(f"❌ Fichier manquant : {X_path}")
else:
    print(f"✔ Fichier trouvé : {X_path}")

if not os.path.exists(y_path):
    print(f"❌ Fichier manquant : {y_path}")
else:
    print(f"✔ Fichier trouvé : {y_path}")

# Charger un échantillon pour vérification
print("\n=== Chargement des fichiers pour vérification ===")
X_sample = pd.read_csv(X_path, nrows=1000)
y_sample = pd.read_csv(y_path, nrows=1000)

# Vérifier les dimensions
X_rows, X_cols = X_sample.shape
y_rows, y_cols = y_sample.shape

print(f"X_global (sample) : {X_rows} lignes × {X_cols} colonnes")
print(f"y_global (sample) : {y_rows} lignes × {y_cols} colonne(s)")

# Vérifier correspondance nombre de lignes
if X_rows == y_rows:
    print("✔ Le nombre de lignes correspond entre X et y")
else:
    print("⚠ Attention : nombre de lignes différent entre X et y")

# Vérifier NaN ou Inf dans y
if y_sample.isna().sum().sum() == 0 and (y_sample == float('inf')).sum().sum() == 0:
    print("✔ Pas de NaN ou Inf dans y_global")
else:
    print("⚠ Attention : NaN ou Inf détectés dans y_global")

# Aperçu des labels uniques
print(f"\nExemple de labels uniques dans y_global : {y_sample.iloc[:,0].unique()}")
