import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the cleaned dataset
print("Loading cleaned dataset...")
try:
    df = pd.read_csv('cleaned_Viral_Social_Media_Trends.csv')
    print(f"Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("Error: 'cleaned_Viral_Social_Media_Trends.csv' not found. Please ensure the file exists.")
    exit()

# 2. Split the dataset into training and testing sets (80:20 ratio)
print("\nSplitting dataset into training and testing sets...")
# Use stratified split on 'Platform' if it exists to maintain distribution
if 'Platform' in df.columns:
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['Platform']
    )
else:
    print("Warning: 'Platform' column not found. Using random split without stratification.")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

# Print sizes of the resulting sets
print(f"Training set: {train_df.shape[0]} rows ({train_df.shape[0]/df.shape[0]*100:.1f}%)")
print(f"Testing set: {test_df.shape[0]} rows ({test_df.shape[0]/df.shape[0]*100:.1f}%)")

# 3. Save the training and testing sets
train_df.to_csv('training_set.csv', index=False)
test_df.to_csv('testing_set.csv', index=False)
print("\nSaved training set as 'training_set.csv'")
print("Saved testing set as 'testing_set.csv'")

# 4. Document the splitting process
print("\nDocumenting splitting process...")
with open('data_splitting_report.txt', 'w') as f:
    f.write("Data Splitting Report:\n")
    f.write(f"- Loaded cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")
    f.write(f"- Split ratio: 80% training, 20% testing\n")
    f.write(f"- Training set: {train_df.shape[0]} rows ({train_df.shape[0]/df.shape[0]*100:.1f}%)\n")
    f.write(f"- Testing set: {test_df.shape[0]} rows ({test_df.shape[0]/df.shape[0]*100:.1f}%)\n")
    f.write("- Stratification: Applied on 'Platform' column if available, otherwise random split\n")
print("Data splitting report saved as 'data_splitting_report.txt'")

print("\nDataset splitting completed!")