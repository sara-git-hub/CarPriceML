import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
OUTPUT_DIR = 'visualizations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filepath='../data/car-details.csv'):
    """Charger les données"""
    df = pd.read_csv(filepath)
    print(f"\n✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"Colonnes : {', '.join(df.columns)}\n")
    return df

def show_basic_info(df):
    """Afficher les infos générales et statistiques"""
    print("=== INFORMATIONS GÉNÉRALES ===")
    print(df.info())
    print("\n=== VALEURS MANQUANTES ===")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("Aucune valeur manquante")
    else:
        print(missing)
    print(f"\nDoublons : {df.duplicated().sum()}")

    print("\n=== STATISTIQUES NUMÉRIQUES ===")
    print(df.describe())

def analyze_categories(df):
    """Afficher les distributions des variables catégorielles"""
    print("\n=== VARIABLES CATÉGORIELLES ===")
    for col in df.select_dtypes(include='object'):
        print(f"\n{col} : {df[col].nunique()} valeurs uniques")
        print(df[col].value_counts().head(5))

def detect_outliers(df):
    """Détection simple des outliers par IQR"""
    print("\n=== OUTLIERS (IQR) ===")
    for col in df.select_dtypes(include=np.number):
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_out > 0:
            print(f"{col}: {n_out} ({n_out/len(df)*100:.2f}%)")
    print("")

def correlation_summary(df, target='selling_price'):
    """Affiche les corrélations avec la variable cible"""
    num_cols = df.select_dtypes(include=np.number)
    if target not in num_cols: return
    corr = num_cols.corr()[target].sort_values(ascending=False)
    print("\n=== CORRÉLATIONS AVEC LE PRIX ===")
    print(corr)

def save_plot(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{name}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ {name}.png")

def create_visuals(df):
    """Création rapide des graphiques principaux"""
    print("\n=== VISUALISATIONS ===")

    if 'selling_price' in df:
        # 1. Distribution du prix
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(df['selling_price'], bins=40, kde=True, ax=ax[0], color='skyblue')
        sns.boxplot(y=df['selling_price'], ax=ax[1], color='coral')
        ax[0].set_title('Distribution du prix'); ax[1].set_title('Boxplot du prix')
        save_plot(fig, '01_price_distribution')

    # 2. Prix moyen par marque
    if {'company', 'selling_price'}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(10, 6))
        top = df['company'].value_counts().head(10).index
        sns.barplot(x='selling_price', y='company', data=df[df['company'].isin(top)],
                    estimator=np.mean, order=top, ax=ax)
        ax.set_title('Prix moyen par marque (Top 10)')
        save_plot(fig, '02_price_by_company')

    # 3. Prix vs Kilométrage
    if {'km_driven', 'selling_price'}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x='km_driven', y='selling_price', data=df, alpha=0.6)
        ax.set_title('Prix vs Kilométrage')
        save_plot(fig, '03_price_vs_kms')

    # 4. Prix moyen par année
    if {'year', 'selling_price'}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        yearly = df.groupby('year')['selling_price'].mean()
        sns.lineplot(x=yearly.index, y=yearly.values, marker='o', ax=ax)
        ax.set_title('Évolution du prix moyen par année')
        save_plot(fig, '04_price_by_year')

    # 5. Répartition du carburant
    if 'fuel' in df:
        fig, ax = plt.subplots(figsize=(6, 6))
        df['fuel'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, cmap='Set2')
        ax.set_ylabel('')
        ax.set_title('Répartition du type de carburant')
        save_plot(fig, '05_fuel_type_distribution')

    # 6. Heatmap de corrélation
    num_cols = df.select_dtypes(include=np.number)
    if len(num_cols.columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(num_cols.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title('Matrice de corrélation')
        save_plot(fig, '06_correlation_heatmap')

def main():
    print("\n" + "="*60)
    print("🚗 ANALYSE EXPLORATOIRE - CarPriceML")
    print("="*60)

    df = load_data()
    show_basic_info(df)
    analyze_categories(df)
    detect_outliers(df)
    correlation_summary(df)
    create_visuals(df)

    print("\n✅ ANALYSE TERMINÉE")
    print(f"📁 Visualisations enregistrées dans '{OUTPUT_DIR}/'\n")

if __name__ == "__main__":
    main()
