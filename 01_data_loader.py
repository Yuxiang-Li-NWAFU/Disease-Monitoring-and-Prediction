import sys
import io
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def ensure_result_dir():
    result_dir = os.path.join(PROJECT_DIR, 'results', '01_split')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    else:
        pass
    return result_dir

def load_and_process_data():
    try:

        result_dir = ensure_result_dir()

        file_paths = [os.path.join(PROJECT_DIR, 'data', 'raw_data.xlsx')]

        data = None
        for path in file_paths:
            try:
                
                data = pd.read_excel(path, sheet_name='Sheet1')
                
                
                break
            except Exception as e:
                print(f"Failed to load data: {str(e)[:50]}...")

        if data is None:
            raise ValueError("Input data file not found")

        
        

        target_col = 'disease_level'
        if target_col not in data.columns:

            possible_targets = [col for col in data.columns if 'level' in col.lower() or 'disease' in col.lower()]
            if possible_targets:
                target_col = possible_targets[0]
                print(f"\nWarning: using '{target_col}' as the target column")
            else:
                raise ValueError("Error: target column 'disease_level' was not found")

        

        data = data.dropna(axis=1, how='all')
        

        encoding_mapping = {}
        mapping_dfs = []

        if 'CITY' in data.columns:
            
            cities = data['CITY'].unique()

            city_mapping = {city: idx for idx, city in enumerate(cities)}
            encoding_mapping['CITY'] = city_mapping

            city_df = pd.DataFrame({
                'Original city': list(city_mapping.keys()),
                'Code': list(city_mapping.values())
            })
            city_df = city_df.sort_values('Code').reset_index(drop=True)
            mapping_dfs.append(('City mapping', city_df))

            data['CITY'] = data['CITY'].map(city_mapping)
            

        if 'county' in data.columns:
            
            counties = data['county'].unique()

            county_mapping = {county: idx for idx, county in enumerate(counties)}
            encoding_mapping['county'] = county_mapping

            county_df = pd.DataFrame({
                'Original county': list(county_mapping.keys()),
                'Code': list(county_mapping.values())
            })
            county_df = county_df.sort_values('Code').reset_index(drop=True)
            mapping_dfs.append(('County mapping', county_df))

            data['county'] = data['county'].map(county_mapping)
            

        imputer = SimpleImputer(strategy='mean')
        cols_to_impute = [col for col in data.columns if
                          col not in ['CITY', 'county', target_col] and col in data.columns]
        if cols_to_impute:
            data[cols_to_impute] = imputer.fit_transform(data[cols_to_impute])
            

        
        label_counts = Counter(data[target_col])
        

        min_samples = 2
        classes_to_keep = [cls for cls, count in label_counts.items() if count >= min_samples]
        mask = data[target_col].isin(classes_to_keep)

        before_count = len(data)
        data = data.loc[mask]
        after_count = len(data)

        
        

        columns_to_drop = [target_col]
        if 'year' in data.columns:
            columns_to_drop.append('year')
            

        columns_to_drop = [col for col in columns_to_drop if col in data.columns]

        X = data.drop(columns=columns_to_drop)
        y = data[target_col]

        
        test_size = 0.2
        if len(data) < 100:
            test_size = min(0.2, 5 / len(data))
            

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size,
            random_state=42, stratify=y
        )

        print(f"Training-set size: {len(X_train)} samples")
        print(f"Test-set size: {len(X_test)} samples")

        
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        train_path = os.path.join(result_dir, 'train_data_processed.csv')
        test_path = os.path.join(result_dir, 'test_data_processed.csv')
        mapping_path = os.path.join(result_dir, 'encoding_mapping.xlsx')
        summary_path = os.path.join(result_dir, 'data_summary.txt')

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        
        

        if mapping_dfs:
            with pd.ExcelWriter(mapping_path) as writer:
                for sheet_name, df_map in mapping_dfs:
                    df_map.to_excel(writer, sheet_name=sheet_name, index=False)
            

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Raw data shape: {before_count} rows x {data.shape[1]} columns\n")
            f.write(f"Processed data shape: {after_count} rows x {data.shape[1]} columns\n")
            f.write(f"Training-set size: {len(X_train)} samples\n")
            f.write(f"Test-set size: {len(X_test)} samples\n")
            f.write("\nTarget distribution:\n")
            for cls, count in sorted(label_counts.items()):
                f.write(f"  {cls}: {count} samples\n")

            if 'CITY' in encoding_mapping:
                f.write("\nCity encoding map:\n")
                for city, code in encoding_mapping['CITY'].items():
                    f.write(f"  {city}: {code}\n")

            if 'county' in encoding_mapping:
                f.write("\nCounty encoding map:\n")
                for county, code in encoding_mapping['county'].items():
                    f.write(f"  {county}: {code}\n")

        print(f"\nResults saved to {result_dir}.")
        return True

    except Exception as e:
        print(f"\nData processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = load_and_process_data()
    if not success:
        print("Data processing failed. Check the error log.")
