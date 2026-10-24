import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import os
import warnings
import shutil

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

warnings.filterwarnings("ignore", category=UserWarning)

def preprocess_data():

    result_dir = os.path.join(PROJECT_DIR, 'results', '02_preprocessed')
    os.makedirs(result_dir, exist_ok=True)

    train_file = os.path.join(PROJECT_DIR, 'results', '01_split', 'train_data_processed.csv')
    test_file = os.path.join(PROJECT_DIR, 'results', '01_split', 'test_data_processed.csv')

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    if 'disease_level' in train_data.columns:
        X_train = train_data.drop('disease_level', axis=1)
        y_train = train_data['disease_level']
    else:
        raise KeyError("The training data do not contain the 'disease_level' column")

    if 'disease_level' in test_data.columns:
        X_test = test_data.drop('disease_level', axis=1)
        y_test = test_data['disease_level']
    else:
        raise KeyError("The test data do not contain the 'disease_level' column")

    

    

    y_train_binary = y_train.apply(lambda x: 0 if x <= 2 else 1)
    y_test_binary = y_test.apply(lambda x: 0 if x <= 2 else 1)

    
    
    
    

    columns_to_remove = ['CITY', 'county', 'City', 'country', 'Country']
    columns_to_keep = [col for col in X_train.columns if col not in columns_to_remove]

    
    

    X_train = X_train[columns_to_keep]
    X_test = X_test[columns_to_keep]

    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    
    

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    try:

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )
    except TypeError:

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='passthrough'
        )

    
    preprocessor.fit(X_train)

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Preprocessed training-set shape: {X_train_processed.shape}")
    print(f"Preprocessed test-set shape: {X_test_processed.shape}")

    try:
        feature_names = preprocessor.get_feature_names_out()
        
    except (AttributeError, NotImplementedError):
        

        feature_names = []

        for name, transformer, cols in preprocessor.transformers_:
            if name == 'num':

                feature_names.extend(cols)

        remainder_indices = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'remainder':
                if hasattr(cols, '__iter__') and not isinstance(cols, str):
                    remainder_indices = cols
                break

        if remainder_indices:
            
            try:

                remainder_names = [X_train.columns[i] for i in remainder_indices]
            except Exception:

                remainder_names = [f"remainder_{i}" for i in range(len(remainder_indices))]

            feature_names.extend(remainder_names)

    if not hasattr(preprocessor, 'verbose_feature_names_out') or (
            hasattr(preprocessor, 'verbose_feature_names_out') and preprocessor.verbose_feature_names_out):
        
        feature_names = [name.replace('num__', '').replace('remainder__', '') for name in feature_names]

    print(f"Number of features: {len(feature_names)}")
    

    train_save_path = os.path.join(result_dir, 'train_preprocessed.csv')
    test_save_path = os.path.join(result_dir, 'test_preprocessed.csv')
    features_save_path = os.path.join(result_dir, 'feature_names.csv')
    preprocessor_save_path = os.path.join(result_dir, 'preprocessor.pkl')

    def safe_save(data, path, header=None):

        temp_path = path + ".tmp"
        try:

            np.savetxt(temp_path, data, delimiter=',', header=header, comments='')

            if os.path.exists(path):
                try:
                    os.remove(path)
                except PermissionError:
                    print(f"Warning: could not remove the existing file {path}. It may be open in another application.")
                    
                    raise

            os.rename(temp_path, path)
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    try:

        train_data_to_save = np.c_[X_train_processed, y_train_binary.values]
        safe_save(train_data_to_save, train_save_path,
                  header=','.join(list(feature_names) + ['disease_level_binary']))

        test_data_to_save = np.c_[X_test_processed, y_test_binary.values]
        safe_save(test_data_to_save, test_save_path,
                  header=','.join(list(feature_names) + ['disease_level_binary']))
    except PermissionError:
        print("\nError: unable to write the file; it may be open in another application.")
        
        
        
        
        return

    pd.Series(feature_names).to_csv(features_save_path, index=False)
    

    joblib.dump(preprocessor, preprocessor_save_path)
    

    print(f"Preprocessing complete. All outputs are saved in [{result_dir}].")
    return X_train_processed, y_train_binary.values, X_test_processed, y_test_binary.values, feature_names

if __name__ == "__main__":
    preprocess_data()
