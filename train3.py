import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, MinMaxScaler

num_features = [] 
fl_features = [] 
cat_features = []

def load_data(nom_fichier, delimiter=';', encoding='latin1'):
    data = pd.read_csv(nom_fichier, encoding=encoding, delimiter=delimiter)
    print(f"Contenu lu du fichier {nom_fichier} :\n{data}")
    return data

def remove_outliers_iqr(data, features, max_outliers):
    initial_rows = data.shape[0]
    Q1 = data[features].quantile(0.25)
    Q3 = data[features].quantile(0.75)
    IQR = Q3 - Q1
    approx = 1.5
    outliers_indices = ((data[features] < (Q1 - approx * IQR)) | (data[features] > (Q3 + approx * IQR))).any(axis=1)
    if max_outliers is not None and outliers_indices.sum() > max_outliers:
        top_outliers = outliers_indices.nlargest(max_outliers).index
        outliers_indices.loc[top_outliers] = False
    print("Enregistrments Avant TRT des Outliers = ", initial_rows)
    print("Enregistrements restants              = ", len(data[~outliers_indices]))
    print("Outliers supprimés                    = ", len(data[outliers_indices]))
    return data[~outliers_indices]

def display_column_names(data):
    colonnes = data.columns
    print("Noms de colonnes du dataset :")
    for colonne in colonnes:
        print(colonne)

def define_features_to_use():
    global num_features, fl_features, cat_features
    num_features = ["total_area_sqm", "surface_land_sqm", "nbr_bedrooms", "nbr_frontages", 
                    "garden_sqm", "terrace_sqm", "construction_year"]
    fl_features = ["fl_terrace", "fl_garden", "fl_furnished", "fl_double_glazing",
                   "fl_swimming_pool", "fl_open_fire", "fl_floodzone"]
    cat_features = ["property_type", "subproperty_type", "region", "province", "locality", 
                    "state_building", "equipped_kitchen", "heating_type"]

def export_file_for_predictions(data):
    df_export_all_features_predict = pd.DataFrame(data[num_features + fl_features + cat_features])
    df_export_all_features_predict_non_empty = df_export_all_features_predict[df_export_all_features_predict.notna().all(axis=1)].head(10)
    df_export_all_features_predict_non_empty.to_csv(r'absolute/path/to/data/input.csv', sep=';', index=False)

def split_data_into_features_and_target(data):
    X = data[num_features + fl_features + cat_features]
    y = data["price"]
    return X, y

def prepare_data(data):
    features_for_outliers = ["total_area_sqm", "surface_land_sqm", "nbr_bedrooms", "nbr_frontages", 
                             "garden_sqm", "terrace_sqm", "construction_year"]
    max_outliers_to_remove = 1000
    data = remove_outliers_iqr(data, features_for_outliers, max_outliers=max_outliers_to_remove)
    return split_data_into_features_and_target(data)

def train():
    data = load_data(r'data\properties.csv')
    display_column_names(data)
    define_features_to_use()
    export_file_for_predictions(data)
    
    X, y = prepare_data(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=505)

    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])

    enc = OneHotEncoder()
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    X_train = pd.concat(
        [X_train[num_features + fl_features].reset_index(drop=True),
         pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out())],
        axis=1,
    )

    X_test = pd.concat(
        [X_test[num_features + fl_features].reset_index(drop=True),
         pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out())],
        axis=1,
    )

    feature_scaler = MinMaxScaler()
    X_train[num_features] = feature_scaler.fit_transform(X_train[num_features])
    X_test[num_features] = feature_scaler.transform(X_test[num_features])

    degree = 1
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train)

    train_score_poly = r2_score(y_train, model_poly.predict(X_train_poly))
    test_score_poly = r2_score(y_test, model_poly.predict(X_test_poly))
    print(f"Train R² score (polynomial): {train_score_poly}")
    print(f"Test R² score (polynomial): {test_score_poly}")

if __name__ == "__main__":
    train()
