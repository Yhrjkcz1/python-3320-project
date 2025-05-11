import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import os
# 这个Model中的engagement level是重新计算然后替换原本的值
# Suppress TensorFlow INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class EngagementClassifier2:
    def __init__(self):
        self.model = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.train_df = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.history = {}  # Dictionary to store model histories
        self.metrics = {}  # Dictionary to store metrics (accuracy, precision, etc.)
        self.categorical_cols = ['Platform', 'Hashtag', 'Content_Type', 'Region']
        self.numerical_cols = ['Views', 'Likes', 'Shares', 'Comments']

    def read_dataset(self):
        """Read the dataset from a CSV file."""
        self.train_df = pd.read_csv('Viral_Social_Media_Trends.csv')

    def calculate_engagement_level(self, row):
        """Calculate Engagement_Level based on the provided formula."""
        # Handle division by zero
        if row['Views'] == 0:
            return 'Low'  # Default to Low if Views is zero to avoid division by zero
        engagement_ratio = (row['Likes'] + row['Shares'] + row['Comments']) / row['Views']
        if engagement_ratio > 0.1:
            return 'High'
        elif 0.05 <= engagement_ratio <= 0.1:
            return 'Medium'
        else:
            return 'Low'

    def preprocess_data(self):
        """Preprocess the dataset: calculate new Engagement_Level, handle missing values, encode categorical variables, and scale numerical features."""
        # Drop unnecessary column
        self.train_df = self.train_df.drop('Post_ID', axis=1)

        # Calculate new Engagement_Level based on the formula, overriding any existing values
        self.train_df['Engagement_Level'] = self.train_df.apply(self.calculate_engagement_level, axis=1)
        print("New Engagement_Level distribution:\n", self.train_df['Engagement_Level'].value_counts())

        # Handle missing values (if any)
        self.train_df[self.numerical_cols] = self.train_df[self.numerical_cols].fillna(self.train_df[self.numerical_cols].mean())
        self.train_df[self.categorical_cols] = self.train_df[self.categorical_cols].fillna(self.train_df[self.categorical_cols].mode().iloc[0])

        # Encode categorical variables (one-hot encoding)
        self.train_df = pd.get_dummies(self.train_df, columns=self.categorical_cols, drop_first=True)

        # Encode target variable (Engagement_Level: High, Medium, Low)
        self.train_df['Engagement_Level'] = self.label_encoder.fit_transform(self.train_df['Engagement_Level'])

        # Separate features and target
        self.train_x = self.train_df.drop('Engagement_Level', axis=1)
        self.train_y = self.train_df['Engagement_Level']

        # Scale numerical features
        self.train_x[self.numerical_cols] = self.scaler.fit_transform(self.train_x[self.numerical_cols])

        # Split data into training and testing sets
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.train_x, self.train_y, test_size=0.2, random_state=42, stratify=self.train_y
        )

    def build_my_model(self):
        """Build a deeper custom neural network model using Keras with additional layers and Dropout."""
        self.model = Sequential()
        # Input layer
        self.model.add(Input(shape=(self.train_x.shape[1],)))
        # Hidden layers with increasing depth
        self.model.add(Dense(128, activation='relu'))  # First hidden layer with 128 neurons
        self.model.add(Dropout(0.3))  # Dropout to prevent overfitting
        self.model.add(Dense(64, activation='relu'))   # Second hidden layer with 64 neurons
        self.model.add(Dropout(0.2))  # Dropout with reduced rate
        self.model.add(Dense(32, activation='relu'))   # Third hidden layer with 32 neurons
        self.model.add(Dropout(0.2))  # Dropout
        self.model.add(Dense(16, activation='relu'))   # Fourth hidden layer with 16 neurons
        # Output layer (3 classes: High, Medium, Low)
        self.model.add(Dense(3, activation='softmax'))

        # Compile the model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def train_my_model(self):
        """Train the custom neural network model."""
        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
        history = self.model.fit(
            self.train_x, self.train_y, epochs=50, validation_split=0.2, 
            batch_size=32, callbacks=[early_stopping], verbose=1
        )
        self.history['My Model'] = history

        # Evaluate the model
        predicted_probs = self.model.predict(self.test_x)
        predicted_labels = predicted_probs.argmax(axis=1)
        self.calculate_metrics('My Model', predicted_labels)
        return history

    def train_decision_tree(self):
        """Train a Decision Tree classifier."""
        dt_classifier = DecisionTreeClassifier(random_state=42)
        dt_classifier.fit(self.train_x, self.train_y)
        
        # Evaluate and calculate metrics
        predicted_labels = dt_classifier.predict(self.test_x)
        self.calculate_metrics('Decision Tree', predicted_labels)
        return dt_classifier

    def train_random_forest(self):
        """Train a Random Forest classifier."""
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(self.train_x, self.train_y)
        
        # Evaluate and calculate metrics
        predicted_labels = rf_classifier.predict(self.test_x)
        self.calculate_metrics('Random Forest', predicted_labels)
        return rf_classifier

    def train_naive_bayes(self):
        """Train a Naive Bayes classifier."""
        nb_classifier = GaussianNB()
        nb_classifier.fit(self.train_x, self.train_y)
        
        # Evaluate and calculate metrics
        predicted_labels = nb_classifier.predict(self.test_x)
        self.calculate_metrics('Naive Bayes', predicted_labels)
        return nb_classifier

    def calculate_metrics(self, model_name, predicted_labels):
        """Calculate evaluation metrics and confusion matrix for a model."""
        accuracy = accuracy_score(self.test_y, predicted_labels)
        precision = precision_score(self.test_y, predicted_labels, average='weighted')
        recall = recall_score(self.test_y, predicted_labels, average='weighted')
        f1 = f1_score(self.test_y, predicted_labels, average='weighted')
        cm = confusion_matrix(self.test_y, predicted_labels)

        # Store metrics
        self.metrics[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }

    def plot_metrics(self):
        """Plot accuracy and other metrics for all models."""
        model_names = ['My Model', 'Decision Tree', 'Random Forest', 'Naive Bayes']
        accuracies = [self.metrics[model]['accuracy'] for model in model_names]

        # Plot accuracy comparison
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'orange'])
        plt.xlabel('Model')
        plt.ylabel('Test Accuracy')
        plt.title('Model Comparison (Test Accuracy)')
        plt.ylim(0, 1)
        plt.show()

        # Display metrics and plot confusion matrices
        for model_name, model_metrics in self.metrics.items():
            print(f"\nMetrics for {model_name}:")
            print(f"  Accuracy: {model_metrics['accuracy']:.4f}")
            print(f"  Precision: {model_metrics['precision']:.4f}")
            print(f"  Recall: {model_metrics['recall']:.4f}")
            print(f"  F1-Score: {model_metrics['f1_score']:.4f}")

            # Plot confusion matrix
            cm = model_metrics['confusion_matrix']
            plt.figure(figsize=(6, 6))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap='Blues', cbar=False,
                xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_
            )
            plt.title(f'Confusion Matrix for {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()

    def train_all_models(self):
        """Train all models and plot their metrics."""
        print("Preprocessing data...")
        self.preprocess_data()

        print("Training My Model...")
        self.build_my_model()
        self.train_my_model()

        print("Training Decision Tree...")
        self.train_decision_tree()

        print("Training Random Forest...")
        self.train_random_forest()

        print("Training Naive Bayes...")
        self.train_naive_bayes()

        print("Plotting metrics...")
        self.plot_metrics()

   