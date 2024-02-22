import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, MinMaxScaler

# Définition des listes pour stocker les caractéristiques
num_features = [] 
fl_features = [] 
cat_features = []
data = []
X = []
y = []

def load_data(nom_fichier):
    # Charge les données depuis le fichier CSV
    data = pd.read_csv(nom_fichier, encoding='latin1', delimiter=';')
    print(f"Contenu lu du fichier {nom_fichier} :\n{data}")
    return data
    

def remove_outliers_iqr(data, features, max_outliers=None):
    initial_rows = data.shape[0]
    # Calcul de l'IQR pour les caractéristiques spécifiées
    Q1 = data[features].quantile(0.25)
    Q3 = data[features].quantile(0.75)
    IQR = Q3 - Q1
    approx = 1.5  # Valeur approximative pour le seuil (à modifier au besoin)
    outliers_indices = ((data[features] < (Q1 - approx * IQR)) | (data[features] > (Q3 + approx * IQR))).any(axis=1)

    # Sélection des indices des outliers en fonction du nombre maximal autorisé
    print(outliers_indices.sum()) #******************************************************************
    if max_outliers is not None and outliers_indices.sum() > max_outliers:
        top_outliers = outliers_indices.nlargest(max_outliers).index
        outliers_indices.loc[top_outliers] = False

    outliers_removed = initial_rows - data.shape[0]
    print(f"Nombre de valeurs aberrantes supprimées : {outliers_removed}")
    return data[~outliers_indices]



def display_column_names(data):
    # Affiche les noms des colonnes du dataset
    colonnes = data.columns
    print("Noms de colonnes du dataset :")
    for colonne in colonnes:
        print(colonne)
    # NON UTILISÉ dans les caractéristiques
    # et price !!! car c'est la cible
    # zip_code, id, latitude, longitude,  primary_energy_consumption_sqm, epc, cadastral_income    

def define_features_to_use():
    # Définit les caractéristiques à utiliser
    global num_features, fl_features, cat_features  # Utilisation des variables globales pour affecter les listes
    num_features = ["total_area_sqm", "surface_land_sqm",
                    "nbr_bedrooms", "nbr_frontages", 
                    "garden_sqm", "terrace_sqm",  
                    "construction_year"]
    fl_features = ["fl_terrace", "fl_garden",
                   "fl_furnished", "fl_double_glazing",
                   "fl_swimming_pool", "fl_open_fire",
                   "fl_floodzone"]
    cat_features = ["property_type", "subproperty_type", 
                    "region", "province", "locality", 
                    "state_building", 
                    "equipped_kitchen", "heating_type"
                    ]

def export_file_for_predictions(data):
    # take the 10 first element not null in data & Export the file
    df_export_file_predict = pd.DataFrame(data[num_features + fl_features + cat_features])
    df_export_file_predict_non_empty = df_export_file_predict[df_export_file_predict.notna().all(axis=1)].head(10)
    df_export_file_predict_non_empty.to_csv('data\input.csv', sep=';', index=False)

def split_data_into_features_target():
    X = data[num_features + fl_features + cat_features]
    y = data["price"]
    
def split_data_into_training_and_test_sets():
    # Split Features & Target into training & test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    ) 

def train():
    data = load_data("data\properties.csv")
    display_column_names(data)
    define_features_to_use()
    export_file_for_predictions(data)
    # Supprime les valeurs aberrantes en utilisant l'IQR et contrôle le nombre de valeurs aberrantes supprimées
    # Specify with which features, data will be clean
    # Define the maximum of ouliers you agree to delete
    features_for_outliers =  ["total_area_sqm", "surface_land_sqm",
                    "nbr_bedrooms", "nbr_frontages", 
                    "garden_sqm", "terrace_sqm",  
                    "construction_year"]
    max_outliers_to_remove = 100  
    data = remove_outliers_iqr(data, features_for_outliers, max_outliers=max_outliers_to_remove)
    split_data_into_features_target()
    split_data_into_training_and_test_sets()
    

    # Impute les valeurs manquantes en utilisant SimpleImputer
    imputer = SimpleImputer(strategy="mean")

    imputer.fit(X_train[num_features])

    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])

    # Convertit les colonnes catégorielles avec OneHotEncoder
    enc = OneHotEncoder()
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine les colonnes numériques et catégorielles encodées en one-hot
    X_train = pd.concat(
        [
            X_train[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    X_test = pd.concat(
        [
            X_test[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    print(f"Caractéristiques : \n {X_train.columns.tolist()}")

    # Crée MinMaxScaler pour les caractéristiques numériques
    feature_scaler = MinMaxScaler()

    # Ajuste et transforme les caractéristiques numériques pour les ensembles d'entraînement et de test
    X_train[num_features] = feature_scaler.fit_transform(X_train[num_features])
    X_test[num_features] = feature_scaler.transform(X_test[num_features])

    # Spécifie le degré des caractéristiques polynomiales
    degree =  1 # Vous pouvez ajuster cela au besoin

    # Crée PolynomialFeatures
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)

    # Ajuste et transforme les caractéristiques en caractéristiques polynomiales
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    # Entraîne le modèle sur les caractéristiques polynomiales
    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train)


    # Evaluate the model on polynomial features
    train_score_poly = r2_score(y_train, model_poly.predict(X_train_poly))
    test_score_poly = r2_score(y_test, model_poly.predict(X_test_poly))
    print(f"Train R² score (polynomial): {train_score_poly}")
    print(f"Test R² score (polynomial): {test_score_poly}")




if __name__ == "__main__":
    train()
