# ============================================================
# 03_training_cv_leakage_auto.py
# Entraînement ML IDS avec détection et suppression automatique des colonnes suspectes
# ============================================================

import pandas as pd
import numpy as np
import time
import sys
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

print("\n=== Début du script IDS ML avec leakage-auto-check ===")

# ============================================================
# 1. Chargement datasets
# ============================================================
X_path = r"C:\Users\Lenovo\Desktop\Dataset_prepared\X_global.csv"
y_path = r"C:\Users\Lenovo\Desktop\Dataset_prepared\y_global.csv"

print("\n[INFO] Chargement des datasets...")
X = pd.read_csv(X_path)
y_raw = pd.read_csv(y_path, header=None).iloc[:, 0]

# Filtrer uniquement normal / anormal
valid_labels = ['normal', 'anormal']
mask = y_raw.isin(valid_labels)
X = X.loc[mask].reset_index(drop=True)
y_clean = y_raw.loc[mask].reset_index(drop=True)

# Réalignement X / y
min_len = min(len(X), len(y_clean))
if len(X) != len(y_clean):
    print(f"[INFO] Réalignement X et y : {len(X)} -> {min_len}")
    X = X.iloc[:min_len].reset_index(drop=True)
    y_clean = y_clean.iloc[:min_len].reset_index(drop=True)

# Colonnes numériques
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"[INFO] Conversion colonnes non numériques : {non_numeric}")
    for col in non_numeric:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
print("[SUCCESS] X est entièrement numérique ✔")

# Encodage labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_clean)
print(f"[INFO] Labels encodés : {list(le.classes_)} -> {list(range(len(le.classes_)))}")

# Remplir NaN
if X.isna().sum().sum() > 0:
    X = X.fillna(X.mean())

# ============================================================
# 2. Vérification et suppression automatique des colonnes suspectes (target leakage)
# ============================================================
print("\n=== Vérification automatique TARGET LEAKAGE ===")
suspect_cols = []

threshold_f1 = 0.85  # seuil F1 pour suspecter leakage
for col in X.columns:
    try:
        X_col = X[[col]]
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_col, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_train_s, y_train_s)
        f1 = f1_score(y_test_s, rf.predict(X_test_s))
        if f1 > threshold_f1:
            suspect_cols.append((col, f1))
    except Exception as e:
        continue

if suspect_cols:
    print("[WARNING] Colonnes suspectes détectées (possible leakage) :")
    for col, f1 in suspect_cols:
        print(f" - {col} (F1 rapide = {f1:.4f})")
    # Suppression automatique
    X.drop(columns=[c[0] for c in suspect_cols], inplace=True)
    print(f"[INFO] {len(suspect_cols)} colonnes suspectes supprimées.")
else:
    print("[OK] Aucune colonne suspecte détectée ✔")

# Test F1 rapide post-suppression
X_train_small, X_test_small, y_train_small, y_test_small = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)
rf_test = RandomForestClassifier(n_estimators=50, random_state=42)
rf_test.fit(X_train_small, y_train_small)
f1_post = f1_score(y_test_small, rf_test.predict(X_test_small))
print(f"[INFO] F1-score rapide après suppression : {f1_post:.4f}")
if f1_post > 0.98:
    print("🔥 F1 toujours trop élevé → revoir dataset / features 🔥")
else:
    print("✔ Leakage corrigé, métriques plausibles.")

# ============================================================
# 3. Découpage Train/Test
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"[INFO] Train X={X_train.shape}, Test X={X_test.shape}")

# ============================================================
# 4. ROC-AUC robuste
# ============================================================
def compute_roc_auc_safe(model, X_eval, y_eval):
    try:
        return roc_auc_score(y_eval, model.predict_proba(X_eval)[:,1])
    except:
        try:
            return roc_auc_score(y_eval, model.decision_function(X_eval))
        except:
            return roc_auc_score(y_eval, model.predict(X_eval))

# ============================================================
# 5. Evaluation finale
# ============================================================
def evaluate_model_final(name, model, X_te):
    print(f"\n=== Évaluation finale : {name} ===")
    y_pred = model.predict(X_te)
    report = classification_report(y_test, y_pred, target_names=le.classes_, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    roc = compute_roc_auc_safe(model, X_te, y_test)
    print(report)
    print(cm)
    print(f"ROC-AUC = {roc:.4f}")
    return {"Model": name, "ROC-AUC": roc}

# ============================================================
# 6. Entraînement et CV
# ============================================================
models = [
    ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')),
    ("XGBoost", XGBClassifier(n_estimators=250, max_depth=8, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              objective="binary:logistic", tree_method="hist",
                              eval_metric='logloss')),
    ("KNN", KNeighborsClassifier(n_neighbors=5)),
    ("MLP", MLPClassifier(hidden_layer_sizes=(64,32), max_iter=100, random_state=42))
]

results = []
scaler = StandardScaler()

for name, model in models:
    print(f"\n---- Entraînement {name} ----")
    if name == "MLP":
        X_train_model = scaler.fit_transform(X_train)
        X_test_model = scaler.transform(X_test)
    else:
        X_train_model = X_train.values
        X_test_model = X_test.values

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    try:
        cv_scores = cross_val_score(model, X_train_model, y_train, cv=cv, scoring='f1', n_jobs=1)
        print(f"[CV] F1 = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    except Exception as e:
        print(f"[WARNING] CV impossible : {e}")

    model.fit(X_train_model, y_train)
    results.append(evaluate_model_final(name, model, X_test_model))

# ============================================================
# 7. Sauvegarde des performances
# ============================================================
df_results = pd.DataFrame(results)
out_csv = r"C:\Users\Lenovo\Desktop\Dataset_prepared\model_performance_cv_leakage_auto.csv"
df_results.to_csv(out_csv, index=False)
print(f"\n✔ Résultats sauvegardés dans : {out_csv}")
print("\n=== Script terminé ===")
