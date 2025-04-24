import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(input_file='Viral_Social_Media_Trends.csv',  # 原始数据集
                  cleaned_output='cleaned_Viral_Social_Media_Trends.csv',  # 清理后的数据集
                  train_output='training_set.csv', 
                  test_output='testing_set.csv', 
                  report_file='data_splitting_report.txt', 
                  test_size=0.2, 
                  random_state=42):
    """
    Clean the dataset by removing the 'Region' column, save the cleaned data, and split it into training and testing sets.
    
    Args:
        input_file (str): Path to the original dataset CSV.
        cleaned_output (str): Path to save the cleaned dataset CSV.
        train_output (str): Path to save the training set CSV.
        test_output (str): Path to save the testing set CSV.
        report_file (str): Path to save the splitting report.
        test_size (float): Proportion of the dataset for testing (default: 0.2).
        random_state (int): Seed for reproducibility (default: 42).
    
    Returns:
        None: Saves cleaned dataset, training/testing sets, and report to files.
    """
    # Load the original dataset
    print("Loading original dataset...")
    try:
        df = pd.read_csv(input_file)
        print(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        print("Columns in dataset:", df.columns.tolist())  # 打印列名以调试
    except FileNotFoundError:
        print(f"Error: '{input_file}' not found. Please ensure the file exists.")
        return

    # Remove 'Region' column if it exists (注意大小写)
    if 'Region' in df.columns:
        df = df.drop('Region', axis=1)
        print("Removed 'Region' column.")
    else:
        print("Warning: 'Region' column not found in the dataset. Available columns:", df.columns.tolist())

    # Save the cleaned dataset
    df.to_csv(cleaned_output, index=False)
    print(f"Saved cleaned dataset as '{cleaned_output}'")

    # Verify the saved file
    try:
        saved_df = pd.read_csv(cleaned_output)
        print(f"Verified: Cleaned dataset loaded from '{cleaned_output}' has {saved_df.shape[1]} columns")
        print("Columns in cleaned dataset:", saved_df.columns.tolist())  # 打印保存后的列名
    except Exception as e:
        print(f"Error verifying saved file: {e}")

    # Split the dataset into training and testing sets
    print("\nSplitting dataset into training and testing sets...")
    if 'Platform' in df.columns:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['Platform']
        )
        stratification = "Applied on 'Platform' column"
    else:
        print("Warning: 'Platform' column not found. Using random split without stratification.")
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )
        stratification = "Random split (no stratification)"

    # Print sizes of the resulting sets
    print(f"Training set: {train_df.shape[0]} rows ({train_df.shape[0]/df.shape[0]*100:.1f}%)")
    print(f"Testing set: {test_df.shape[0]} rows ({test_df.shape[0]/df.shape[0]*100:.1f}%)")

    # Save the training and testing sets
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    print(f"\nSaved training set as '{train_output}'")
    print(f"Saved testing set as '{test_output}'")

    # Document the splitting process
    print("\nDocumenting splitting process...")
    with open(report_file, 'w') as f:
        f.write("Data Splitting Report:\n")
        f.write(f"- Loaded original dataset: {df.shape[0]} rows, {df.shape[1]+1 if 'Region' in pd.read_csv(input_file).columns else df.shape[1]} columns\n")
        f.write(f"- Cleaned dataset (after removing 'Region'): {df.shape[0]} rows, {df.shape[1]} columns\n")
        f.write(f"- Split ratio: {int((1-test_size)*100)}% training, {int(test_size*100)}% testing\n")
        f.write(f"- Training set: {train_df.shape[0]} rows ({train_df.shape[0]/df.shape[0]*100:.1f}%)\n")
        f.write(f"- Testing set: {test_df.shape[0]} rows ({test_df.shape[0]/df.shape[0]*100:.1f}%)\n")
        f.write(f"- Stratification: {stratification}\n")
    print(f"Data splitting report saved as '{report_file}'")

    print("\nDataset cleaning and splitting completed!")

# 调用函数
split_dataset()