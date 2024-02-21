import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def remove_outliers_iqr(data, features, max_outliers=None):
    # Calcul de l'IQR pour les caractéristiques spécifiées
    Q1 = data[features].quantile(0.25)
    Q3 = data[features].quantile(0.75)
    IQR = Q3 - Q1
    
    approx = 1.r2_score   #normaly 1.5
    # Détermination des indices des outliers pour chaque caractéristique
    outliers_indices = ((data[features] < (Q1 - approx * IQR)) | (data[features] > (Q3 + approx * IQR))).any(axis=1)

    # Sélection des indices des outliers en fonction du nombre maximal autorisé
    if max_outliers is not None and outliers_indices.sum() > max_outliers:
        top_outliers = outliers_indices.nlargest(max_outliers).index
        outliers_indices.loc[top_outliers] = False

    return data[~outliers_indices]

def train():
    # Load the data
    nom_fichier = "data\properties.csv"
    data = pd.read_csv(nom_fichier, encoding='latin1', delimiter=';')
    print(f"Contenu lu du fichier {nom_fichier} :\n{data}")

    # Define features to use
    num_features = ["construction_year", "total_area_sqm", "surface_land_sqm", "nbr_bedrooms", "nbr_frontages", "cadastral_income", "garden_sqm", "terrace_sqm"]
    fl_features = ["fl_terrace", "fl_double_glazing", "fl_furnished", "fl_open_fire", "fl_garden", "fl_swimming_pool", "fl_floodzone"]
    cat_features = ["property_type", "subproperty_type", "region", "province", "locality", "equipped_kitchen", "state_building", "epc", "heating_type"]

    #num_features = ["construction_year", "total_area_sqm"]
    #fl_features = ["fl_terrace", "fl_double_glazing"]
    #cat_features = ["property_type", "subproperty_type"]
    
    
    #num_features = ["surface_land_sqm"]
    #fl_features = ["fl_terrace"]
    #cat_features = ["property_type"]
    
    
    
    export_all_features_predict =  data[num_features + fl_features + cat_features]
    df_export_all_features_predict = pd.DataFrame(export_all_features_predict)
    df_export_all_features_predict_non_empty = df_export_all_features_predict[df_export_all_features_predict.notna().all(axis=1)].head(10)
    
    #Print fichier pour simuler un fichier pour prédictions
    df_export_all_features_predict_non_empty.to_csv('data\input.csv', sep=';', index=False)


    # Specify features for outlier removal
    features_for_outliers = ["construction_year", "total_area_sqm", "surface_land_sqm"]  # Adjust as needed

    # Combine features
    all_features = num_features + fl_features + cat_features + ["price"]

    # Remove outliers using IQR and control the number of outliers removed
    initial_rows = data.shape[0]
    max_outliers_to_remove = 100  # Set your desired maximum number of outliers to remove
    data = remove_outliers_iqr(data, features_for_outliers, max_outliers=max_outliers_to_remove)
    outliers_removed = initial_rows - data.shape[0]
    print(f"Number of outliers removed: {outliers_removed}")
    
    
    

#*************************************************************************************************    
    
    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Treat missing values in "surface_land_sqm" using values from "total_area_sqm"
    #X["surface_land_sqm"].fillna(X["total_area_sqm"], inplace=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])

    # Convert categorical columns with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder()
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
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

    print(f"Features: \n {X_train.columns.tolist()}")

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")

    # Save the model
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "imputer": imputer,
        "enc": enc,
        "model": model,
    }
    joblib.dump(artifacts, "models/artifacts.joblib")


if __name__ == "__main__":
    train()
