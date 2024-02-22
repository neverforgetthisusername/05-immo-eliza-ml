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

def load_data(nom_fichier):
    # Load the data
    data = pd.read_csv(nom_fichier, encoding='latin1', delimiter=';')
    print(f"Contenu lu du fichier {nom_fichier} :\n{data}")
    return data
    

def remove_outliers_iqr(data, features, max_outliers=None):
    # Calcul de l'IQR pour les caractéristiques spécifiées
    Q1 = data[features].quantile(0.25)
    Q3 = data[features].quantile(0.75)
    IQR = Q3 - Q1
    approx = 1.3                                                                # A MODIFIER 

    outliers_indices = ((data[features] < (Q1 - approx * IQR)) | (data[features] > (Q3 + approx * IQR))).any(axis=1)

    # Sélection des indices des outliers en fonction du nombre maximal autorisé
    if max_outliers is not None and outliers_indices.sum() > max_outliers:
        top_outliers = outliers_indices.nlargest(max_outliers).index
        outliers_indices.loc[top_outliers] = False

    return data[~outliers_indices]

def display_column_names(data):
    # Display Colunms Names of the dataset
    colonnes = data.columns
    print("Noms de colonnes du dataset :")
    for colonne in colonnes:
        print(colonne)
    # NOT USED in the Features
    # and price !!! because is the taregt
    # zip_code, id, latitude, longitude,  primary_energy_consumption_sqm, epc, cadastral_income    

def define_features_to_use():
    # Define features to use
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
    # Record csv file for future predictions
    # take the 10 fisrt lines with no empty fields
    df_export_all_features_predict = pd.DataFrame(data[num_features + fl_features + cat_features])
    df_export_all_features_predict_non_empty = df_export_all_features_predict[df_export_all_features_predict.notna().all(axis=1)].head(10)
    df_export_all_features_predict_non_empty.to_csv('data\input.csv', sep=';', index=False)
    
    
def train():
    data = load_data("data\properties.csv")                                     #LOAD THE DATA
    display_column_names(data)                                                  #DISPLAY ALL COLUMNS YOU CAN USE 
    define_features_to_use()                                                    #CHOOSE FEATURES THA YOU WANT TO USE FOR REGRESSION
    export_file_for_predictions(data)                                           #CREATE & EXPORT FILE TO MAKE FUTURE PREDICTIONS

                                
    features_for_outliers = ["total_area_sqm", "surface_land_sqm",             
                             "nbr_bedrooms", "nbr_frontages", 
                             "garden_sqm", "terrace_sqm", 
                             "construction_year"]  # Adjust as needed
    # Remove outliers using IQR and control the number of outliers removed
    initial_rows = data.shape[0]
    max_outliers_to_remove = 100  # Set your desired maximum number of outliers to remove
    data = remove_outliers_iqr(data, features_for_outliers, max_outliers=max_outliers_to_remove)
    outliers_removed = initial_rows - data.shape[0]
    print("là*********************************************************************************")
    print(f"Number of outliers removed: {outliers_removed}")
    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    print("jusqu ici tout va bien")
    imputer.fit(X_train[num_features])
    print("jusqu ici tout va bien")
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

    # Create MinMaxScaler for numerical features
    feature_scaler = MinMaxScaler()

    # Fit and transform numerical features for both training and test sets
    X_train[num_features] = feature_scaler.fit_transform(X_train[num_features])
    X_test[num_features] = feature_scaler.transform(X_test[num_features])

    # Specify the degree of the polynomial features
    degree = 1  # You can adjust this as needed

    # Create PolynomialFeatures
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)

    # Fit and transform the features to polynomial features
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    # Train the model on polynomial features
    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train)

    # Evaluate the model on polynomial features
    train_score_poly = r2_score(y_train, model_poly.predict(X_train_poly))
    test_score_poly = r2_score(y_test, model_poly.predict(X_test_poly))
    print(f"Train R² score (polynomial): {train_score_poly}")
    print(f"Test R² score (polynomial): {test_score_poly}")




if __name__ == "__main__":
    train()
