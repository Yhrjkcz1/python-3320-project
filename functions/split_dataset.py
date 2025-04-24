import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(input_file='cleaned_Viral_Social_Media_Trends.csv', 
                  train_output='training_set.csv', 
                  test_output='testing_set.csv', 
                  report_file='data_splitting_report.txt', 
                  test_size=0.2, 
                  random_state=42):
    """
    Split the cleaned dataset into training and testing sets.
    
    Args:
        input_file (str): Path to the cleaned dataset CSV.
        train_output (str): Path to save the training set CSV.
        test_output (str): Path to save the testing set CSV.
        report_file (str): Path to save the splitting report.
        test_size (float): Proportion of the dataset for testing (default: 0.2).
        random_state (int): Seed for reproducibility (default: 42).
    
    Returns:
        None: Saves training/testing sets and report to files.
    """
    # Load the cleaned dataset
    print("Loading cleaned dataset...")
    try:
        df = pd.read_csv(input_file)
        print(f"Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print(f"Error: '{input_file}' not found. Please ensure the file exists.")
        return

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
        f.write(f"- Loaded cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")
        f.write(f"- Split ratio: {int((1-test_size)*100)}% training, {int(test_size*100)}% testing\n")
        f.write(f"- Training set: {train_df.shape[0]} rows ({train_df.shape[0]/df.shape[0]*100:.1f}%)\n")
        f.write(f"- Testing set: {test_df.shape[0]} rows ({test_df.shape[0]/df.shape[0]*100:.1f}%)\n")
        f.write(f"- Stratification: {stratification}\n")
    print(f"Data splitting report saved as '{report_file}'")

    print("\nDataset splitting completed!")