import pandas as pd
import numpy as np

def knn_classifier(train_file='training_set.csv', 
                   test_file='testing_set.csv', 
                   output_file='test_predictions.csv', 
                   report_file='model_implementation_report.txt', 
                   k=5):
    """
    Implement a k-Nearest Neighbors classifier to predict the best platform.
    
    Args:
        train_file (str): Path to the training set CSV.
        test_file (str): Path to the testing set CSV.
        output_file (str): Path to save predictions CSV.
        report_file (str): Path to save the model report.
        k (int): Number of neighbors for k-NN (default: 5).
    
    Returns:
        None: Saves predictions and report to files.
    """
    # Load training and testing datasets
    print("Loading datasets...")
    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        print(f"Training set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
        print(f"Testing set: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    except FileNotFoundError:
        print(f"Error: '{train_file}' or '{test_file}' not found. Please ensure files exist.")
        return

    # Manual Encoding of Categorical Features
    print("\nEncoding categorical features...")
    feature_cols = ['Hashtag', 'Content_Type', 'Region']
    target_col = 'Best_Platform'

    # Verify required columns exist
    required_cols = feature_cols + [target_col]
    for col in required_cols:
        if col not in train_df.columns or col not in test_df.columns:
            print(f"Error: Column '{col}' missing in dataset. Please check input files.")
            return

    # Frequency encoding for Hashtag
    hashtag_counts = train_df['Hashtag'].value_counts().to_dict()
    train_df['Hashtag_Freq'] = train_df['Hashtag'].map(hashtag_counts)
    test_df['Hashtag_Freq'] = test_df['Hashtag'].map(lambda x: hashtag_counts.get(x, 0))

    # Manual one-hot encoding for Content_Type and Region
    def manual_one_hot_encode(df, column):
        categories = sorted(df[column].unique())
        for category in categories:
            df[f"{column}_{category}"] = (df[column] == category).astype(int)
        return df, categories

    train_df, content_type_cats = manual_one_hot_encode(train_df, 'Content_Type')
    test_df, _ = manual_one_hot_encode(test_df, 'Content_Type')
    train_df, region_cats = manual_one_hot_encode(train_df, 'Region')
    test_df, _ = manual_one_hot_encode(test_df, 'Region')

    # Ensure test set has same one-hot columns as training set
    for col in [f"Content_Type_{cat}" for cat in content_type_cats] + [f"Region_{cat}" for cat in region_cats]:
        if col not in test_df.columns:
            test_df[col] = 0

    # Drop original categorical columns
    train_df = train_df.drop(feature_cols, axis=1)
    test_df = test_df.drop(feature_cols, axis=1)

    # Extract feature columns (all except Best_Platform)
    feature_cols_encoded = [col for col in train_df.columns if col != 'Best_Platform']

    # Convert to numpy arrays for k-NN
    X_train = train_df[feature_cols_encoded].to_numpy()
    y_train = train_df['Best_Platform'].to_numpy()
    X_test = test_df[feature_cols_encoded].to_numpy()

    print(f"Encoded features: {len(feature_cols_encoded)} (Hashtag_Freq + one-hot encoded Content_Type and Region)")

    # Implement k-Nearest Neighbors (k-NN) Classifier
    print("\nImplementing k-NN classifier...")

    def euclidean_distance(x1, x2):
        """Calculate Euclidean distance between two vectors."""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def knn_predict(X_train, y_train, X_test, k):
        """Predict classes for test set using k-NN."""
        predictions = []
        for test_point in X_test:
            distances = [euclidean_distance(test_point, train_point) for train_point in X_train]
            k_indices = np.argsort(distances)[:k]
            k_labels = [y_train[i] for i in k_indices]
            most_common = max(set(k_labels), key=k_labels.count)
            predictions.append(most_common)
        return predictions

    # Make predictions on the test set
    print(f"Running k-NN with k={k}...")
    test_predictions = knn_predict(X_train, y_train, X_test, k=k)

    # Save predictions
    print("\nSaving predictions...")
    test_df['Predicted_Platform'] = test_predictions
    test_df.to_csv(output_file, index=False)
    print(f"Test predictions saved as '{output_file}'")

    # Document the model implementation
    print("\nDocumenting model implementation...")
    with open(report_file, 'w') as f:
        f.write("Model Implementation Report:\n")
        f.write(f"- Loaded datasets: Training ({train_df.shape[0]} rows), Testing ({test_df.shape[0]} rows)\n")
        f.write("- Encoded features:\n")
        f.write("  - Hashtag: Frequency encoding (count of occurrences)\n")
        f.write("  - Content_Type: One-hot encoding\n")
        f.write("  - Region: One-hot encoding\n")
        f.write(f"- Total features: {len(feature_cols_encoded)}\n")
        f.write(f"- Model: k-Nearest Neighbors (k={k}), implemented from scratch\n")
        f.write(f"- Predictions: Generated for test set and saved to '{output_file}'\n")
    print(f"Model implementation report saved as '{report_file}'")

    print("\nModel implementation completed!")