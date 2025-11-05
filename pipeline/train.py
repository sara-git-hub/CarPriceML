import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# === Constantes ===
CONVERSION_RATE = 0.5  # roupie -> MAD
CURRENT_YEAR = 2025
OUTPUT_DIR = "visualizations"
MODEL_DIR = "../models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# === Fonctions principales ===

def load_data(filepath: str) -> pd.DataFrame:
    """Charger et nettoyer les données"""
    print("📥 Chargement des données...")
    df = pd.read_csv(filepath).drop_duplicates().dropna()
    print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def convert_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Convertir la colonne du prix en MAD"""
    df["selling_price"] *= CONVERSION_RATE
    print(f"💰 Prix moyen (MAD) : {df['selling_price'].mean():,.2f}")
    return df


def remove_outliers(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Supprimer les outliers via IQR"""
    print("\n🧹 Suppression des outliers...")
    cols = cols or ["selling_price", "year", "max_power_bhp", "torque_nm"]
    for col in [c for c in cols if c in df.columns]:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    print(f"✅ Après suppression des outliers : {df.shape[0]} lignes restantes")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Créer des variables dérivées"""
    print("\n🧩 Création des features dérivées...")
    if "year" in df.columns:
        df["vehicle_age"] = CURRENT_YEAR - df["year"]
    print("✅ Features dérivées créées.")
    return df


def build_pipeline(num_cols, cat_cols):
    """Créer le pipeline de prétraitement et le modèle"""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])
    base_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
    ])
    # Transformation log sur la variable cible
    return TransformedTargetRegressor(regressor=base_pipeline, func=np.log1p, inverse_func=np.expm1)


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Évaluer le modèle"""
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)

    def metrics(y, y_pred):
        return {
            "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
            "MAE": mean_absolute_error(y, y_pred),
            "R2": r2_score(y, y_pred)
        }

    results = {"train": metrics(y_train, preds_train), "test": metrics(y_test, preds_test)}

    print("\n📊 Performances du modèle :")
    print(f"  🟢 R² (train): {results['train']['R2']:.3f}")
    print(f"  🔵 R² (test) : {results['test']['R2']:.3f}")
    print(f"  📉 RMSE (test): {results['test']['RMSE']:.2f}")
    print(f"  📏 MAE (test) : {results['test']['MAE']:.2f}")

    return results


def plot_feature_importance(model, num_cols, cat_cols):
    """Afficher et sauvegarder le graphique d'importance des features"""
    print("\n📈 Génération du graphique d'importance des features...")

    importances = model.regressor_.named_steps["model"].feature_importances_
    ohe = model.regressor_.named_steps["preprocessor"].named_transformers_["cat"]

    cat_feature_names = ohe.get_feature_names_out(cat_cols) if cat_cols else []
    feature_names = np.concatenate([num_cols, cat_feature_names])

    indices = np.argsort(importances)[::-1]
    feature_names_sorted = np.array(feature_names)[indices]
    importances_sorted = importances[indices]

    top_n = 20
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names_sorted[:top_n][::-1], importances_sorted[:top_n][::-1], color='teal')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Importance des Variables - Random Forest")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
    plt.show()
    print(f"Graphique sauvegardé : {OUTPUT_DIR}/feature_importance.png")


def main():
    """Pipeline complet d'entraînement"""
    print("\n" + "=" * 70)
    print("🚗 DÉMARRAGE DU PIPELINE - CarPriceML")
    print("=" * 70)

    # Chargement et préparation
    df = load_data("../data/car-details.csv")
    df = convert_prices(df)
    df = remove_outliers(df)
    df = create_features(df)

    # Séparation features / target
    #X = df.drop(columns=["selling_price", "name", "seats", "km_driven"], errors="ignore")
    cols=['vehicle_age','year','max_power_bhp','torque_nm','engine_cc']
    X=df[cols]
    y = df["selling_price"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"📊 Jeu d'entraînement : {X_train.shape}, Test : {X_test.shape}")

    # Création et entraînement
    model = build_pipeline(num_cols, cat_cols)
    model.fit(X_train, y_train)

    # Évaluation
    results = evaluate_model(model, X_train, X_test, y_train, y_test)

    # Visualisation des features importantes
    plot_feature_importance(model, num_cols, cat_cols)

    # Sauvegarde
    joblib.dump(model, f"{MODEL_DIR}/rf_model.joblib")
    joblib.dump({"num_cols": num_cols, "cat_cols": cat_cols}, f"{MODEL_DIR}/feature_info.joblib")
    print(f"💾 Modèle sauvegardé dans : {MODEL_DIR}/rf_model.joblib")

    print("\n✅ Pipeline terminé avec succès !")
    return model, results


if __name__ == "__main__":
    main()