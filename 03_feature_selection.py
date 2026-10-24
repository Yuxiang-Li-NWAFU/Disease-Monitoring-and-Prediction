import numpy as np
import pandas as pd
import joblib
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from collections import Counter

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

custom_cmap = LinearSegmentedColormap.from_list("custom", ["#2E86C1", "#E74C3C"])

custom_palette = [custom_cmap(i) for i in np.linspace(0, 1, 10)]

sns.set(style="whitegrid", palette=custom_palette)
sns.set_palette(custom_palette)

def save_data_with_headers(data, feature_names, file_path, label_name='target'):

    columns = list(feature_names) + [label_name]
    df = pd.DataFrame(data, columns=columns)

    df.to_csv(file_path, index=False)
    
    return file_path

def load_data(file_path, is_train=True):

    try:

        data = np.loadtxt(file_path, delimiter=',')

        feature_names = [f"Feature_{i}" for i in range(data.shape[1] - 1)]
        
        return data, feature_names
    except ValueError as ve:

        
        data = pd.read_csv(file_path)

        label_column = 'target' if 'target' in data.columns else data.columns[-1]
        if label_column not in data.columns:

            if data.iloc[:, -1].dtype.kind in 'iufc':
                label_column = data.columns[-1]
                print(f"Column 'target' not found; using the last column '{label_column}' as the target")
            else:

                data['target'] = 0
                label_column = 'target'
                

        feature_names = [col for col in data.columns if col != label_column]

        if label_column != data.columns[-1]:
            cols = feature_names + [label_column]
            data = data[cols]

        X = data[feature_names].values
        y = data[label_column].values.reshape(-1, 1)
        data_array = np.hstack((X, y))

        
        return data_array, feature_names

def combined_filtering(X, y, feature_names, top_n=100):

    
    mi_scores = mutual_info_classif(X, y, random_state=42)

    
    spearman_scores = np.abs([spearmanr(X[:, i], y).correlation
                              for i in range(X.shape[1])])
    spearman_scores = np.nan_to_num(spearman_scores, nan=0.0)

    mi_min, mi_max = mi_scores.min(), mi_scores.max()
    sp_min, sp_max = spearman_scores.min(), spearman_scores.max()

    if mi_max - mi_min < 1e-8:
        mi_normalized = np.zeros_like(mi_scores)
    else:
        mi_normalized = (mi_scores - mi_min) / (mi_max - mi_min + 1e-8)

    if sp_max - sp_min < 1e-8:
        sp_normalized = np.zeros_like(spearman_scores)
    else:
        sp_normalized = (spearman_scores - sp_min) / (sp_max - sp_min + 1e-8)

    combined_scores = (mi_normalized + sp_normalized) / 2

    report = pd.DataFrame({
        'Feature': feature_names,
        'MI_Score': mi_scores,
        'MI_Score_Normalized': mi_normalized,
        'Spearman_Score': spearman_scores,
        'Spearman_Score_Normalized': sp_normalized,
        'Combined_Score': combined_scores
    })

    report_sorted = report.sort_values('Combined_Score', ascending=False)

    top_features = report_sorted.head(top_n)
    top_feature_names = top_features['Feature'].tolist()

    feature_to_index = {name: idx for idx, name in enumerate(feature_names)}
    top_indices = [feature_to_index[name] for name in top_feature_names]

    return X[:, top_indices], top_feature_names, report_sorted

def visualize_selection_process(top_features_report, output_dir):

    try:

        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(16, 10))

        plt.subplot(121)
        top_mi = top_features_report.sort_values('MI_Score', ascending=False).head(30)
        plt.barh(top_mi['Feature'], top_mi['MI_Score'], color=custom_palette[0])
        plt.title('Top Features by Mutual Information')
        plt.xlabel('Mutual Information Score')
        plt.ylabel('Feature')
        plt.gca().invert_yaxis()

        plt.subplot(122)
        top_sp = top_features_report.sort_values('Spearman_Score', ascending=False).head(30)
        plt.barh(top_sp['Feature'], top_sp['Spearman_Score'], color=custom_palette[1])
        plt.title('Top Features by Spearman Correlation')
        plt.xlabel('Spearman Correlation')
        plt.gca().invert_yaxis()

        plt.tight_layout()
        filepath = os.path.join(output_dir, 'top_features_scores.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, min(20, len(top_features_report) * 0.6)))
        top_features_sorted = top_features_report.sort_values('Combined_Score', ascending=True)
        colors = [custom_cmap(score) for score in top_features_sorted['Combined_Score']]
        plt.barh(top_features_sorted['Feature'], top_features_sorted['Combined_Score'],
                 color=colors)
        plt.title('Top Features by Combined Score')
        plt.xlabel('Combined Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'top_features_combined.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.scatter(top_features_report['Spearman_Score'], top_features_report['MI_Score'],
                    c=top_features_report['Combined_Score'], s=100, cmap='viridis', alpha=0.7)
        plt.colorbar(label='Combined Score')
        plt.title('Relationship Between Feature Scores')
        plt.xlabel('Spearman Correlation')
        plt.ylabel('Mutual Information Score')
        plt.grid(True, alpha=0.3)
        filepath = os.path.join(output_dir, 'score_relationship.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 8))
        score_cols = ['MI_Score', 'Spearman_Score', 'Combined_Score']
        score_data = top_features_report[score_cols]
        sns.boxplot(data=score_data, palette=custom_palette[:3])
        plt.title('Distribution of Feature Scores')
        plt.ylabel('Score Value')
        plt.xticks([0, 1, 2], score_cols)
        plt.grid(True, alpha=0.3)
        filepath = os.path.join(output_dir, 'score_distribution.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(18, 12))
        top20 = top_features_report.head(20)
        x = np.arange(len(top20))
        width = 0.25

        plt.bar(x - width, top20['MI_Score'], width, label='MI Score', color=custom_palette[0])
        plt.bar(x, top20['Spearman_Score'], width, label='Spearman Score', color=custom_palette[5])
        plt.bar(x + width, top20['Combined_Score'], width, label='Combined Score', color=custom_palette[9])

        plt.xlabel('Features')
        plt.ylabel('Scores')
        plt.title('Top 20 Features: Detailed Scores')
        plt.xticks(x, top20['Feature'], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filepath = os.path.join(output_dir, 'top20_features_detailed.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(14, 10))
        plt.subplot(211)
        top_norm = top_features_report.head(30)
        plt.barh(top_norm['Feature'], top_norm['MI_Score_Normalized'],
                 color=custom_palette[0], alpha=0.7, label='Normalized MI')
        plt.barh(top_norm['Feature'], top_norm['Spearman_Score_Normalized'],
                 color=custom_palette[1], alpha=0.7, label='Normalized Spearman')
        plt.title('Top Features: Normalized Scores')
        plt.xlabel('Normalized Score')
        plt.ylabel('Feature')
        plt.legend()
        plt.gca().invert_yaxis()

        plt.subplot(212)
        plt.scatter(top_features_report['MI_Score_Normalized'],
                    top_features_report['Spearman_Score_Normalized'],
                    c=top_features_report['Combined_Score'],
                    s=50, cmap='viridis', alpha=0.7)
        plt.colorbar(label='Combined Score')
        plt.title('Normalized Scores Relationship')
        plt.xlabel('Normalized MI Score')
        plt.ylabel('Normalized Spearman Score')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(output_dir, 'normalized_scores_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        
        return output_dir

    except Exception as e:
        print(f"Visualization failed: {str(e)}")
        import traceback
        traceback.print_exc()

def select_features():

    os.makedirs(os.path.join(PROJECT_DIR, 'results', '03_feature_selection', 'data'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_DIR, 'results', '03_feature_selection', 'pictures'), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_DIR, 'results', '03_feature_selection', 'reports'), exist_ok=True)
    
    
    
    

    data_dir = os.path.join(PROJECT_DIR, 'results', '03_feature_selection', 'data')
    picture_dir = os.path.join(PROJECT_DIR, 'results', '03_feature_selection', 'pictures')
    reports_dir = os.path.join(PROJECT_DIR, 'results', '03_feature_selection', 'reports')

    

    train_data, train_feature_names = load_data(os.path.join(PROJECT_DIR, 'results', '02_preprocessed', 'train_preprocessed.csv'), is_train=True)
    test_data, test_feature_names = load_data(os.path.join(PROJECT_DIR, 'results', '02_preprocessed', 'test_preprocessed.csv'), is_train=False)

    if train_data.shape[1] == 0 or test_data.shape[1] == 0:
        raise ValueError("Loaded data are empty. Check the file path and contents.")

    if len(train_feature_names) != train_data.shape[1] - 1:
        print(f"Warning: the number of training features ({len(train_feature_names)}) "
              f"does not match the number of data columns ({train_data.shape[1] - 1}).")
        train_feature_names = [f"Feature_{i}" for i in range(train_data.shape[1] - 1)]

    if len(test_feature_names) != test_data.shape[1] - 1:
        print(f"Warning: the number of test features ({len(test_feature_names)}) "
              f"does not match the number of data columns ({test_data.shape[1] - 1}).")
        test_feature_names = [f"Feature_{i}" for i in range(test_data.shape[1] - 1)]

    if train_feature_names != test_feature_names:
        print("Warning: training and test feature names differ; using the training feature names.")
        test_feature_names = train_feature_names.copy()

    feature_names = train_feature_names
    

    output_path = os.path.join(data_dir, 'train_original.csv')
    save_data_with_headers(train_data, feature_names, output_path, 'target')
    output_path = os.path.join(data_dir, 'test_original.csv')
    save_data_with_headers(test_data, feature_names, output_path, 'target')
    

    
    

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    X_train_selected, selected_feature_names, top_features_report = combined_filtering(
        X_train, y_train, feature_names, top_n=100
    )

    print(f"Number of retained features after selection: {X_train_selected.shape[1]}")

    feature_to_index = {name: idx for idx, name in enumerate(feature_names)}
    selected_indices = [feature_to_index[name] for name in selected_feature_names]

    X_test = test_data[:, :-1]
    X_test_selected = X_test[:, selected_indices]

    y_train_final = train_data[:, -1]
    y_test_final = test_data[:, -1]

    

    output_path = os.path.join(data_dir, 'features_selected_train.csv')
    save_data_with_headers(
        np.c_[X_train_selected, y_train_final],
        selected_feature_names,
        output_path,
        'target'
    )

    output_path = os.path.join(data_dir, 'features_selected_test.csv')
    save_data_with_headers(
        np.c_[X_test_selected, y_test_final],
        selected_feature_names,
        output_path,
        'target'
    )

    output_path = os.path.join(reports_dir, 'selected_features.txt')
    with open(output_path, 'w') as f:
        f.write("\n".join(selected_feature_names))
    

    output_path = os.path.join(reports_dir, 'top_features_report.csv')
    top_features_report.to_csv(output_path, index=False)
    

    
    visualize_selection_process(top_features_report.head(100), os.path.join(picture_dir, 'feature_selection'))

    

    
    for i, feat in enumerate(selected_feature_names):
        row = top_features_report[top_features_report['Feature'] == feat].iloc[0]
        mi_score = row['MI_Score']
        sp_score = row['Spearman_Score']
        mi_norm = row['MI_Score_Normalized']
        sp_norm = row['Spearman_Score_Normalized']
        combined_score = row['Combined_Score']

        
        
        
        

if __name__ == "__main__":

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 200,
    })

    mpl.use('Agg')

    

    start_time = time.time()
    try:
        select_features()
        
    except Exception as e:
        print(f"Main script failed: {str(e)}")
        import traceback

        traceback.print_exc()
        exit(1)
