import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

class SocialMediaEngagementModel:
    def __init__(self, file_path):
        """Initialize with file path and set up attributes."""
        self.file_path = file_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        self.accuracy = None
        self.classification_report = None

    def load_and_process_data(self):
        """Load data and use existing Engagement_Level."""
        self.df = pd.read_csv(self.file_path)
        # Ensure Engagement_Level exists in the dataset
        if 'Engagement_Level' not in self.df.columns:
            raise ValueError("Dataset must contain 'Engagement_Level' column.")

    def check_missing_values(self):
        """Return missing values in the dataset."""
        if self.df is None:
            return "Data not loaded. Call load_and_process_data first."
        return self.df.isnull().sum()

    def check_class_distribution(self):
        """Return class distribution of Engagement_Level."""
        if self.df is None:
            return "Data not loaded. Call load_and_process_data first."
        return self.df['Engagement_Level'].value_counts()

    def check_data_issues(self):
        """Check for potential data issues."""
        if self.df is None:
            return "Data not loaded. Call load_and_process_data first."
        
        issues = []
        
        # Check for missing values in Engagement_Level
        if self.df['Engagement_Level'].isnull().sum() > 0:
            issues.append(f"Found {self.df['Engagement_Level'].isnull().sum()} missing values in Engagement_Level.")
        
        # Check for duplicate rows
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate rows in the dataset.")
        
        return issues if issues else "No obvious data issues detected."

    def encode_features(self):
        """Encode categorical and numerical features."""
        categorical_cols = ['Hashtag', 'Content_Type', 'Region', 'Platform']
        encoded_array = self.encoder.fit_transform(self.df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_array, columns=self.encoder.get_feature_names_out(categorical_cols))

        numerical_features = ['Views']  # Only use Views, exclude Likes, Shares, Comments
        scaled_numerical = self.scaler.fit_transform(self.df[numerical_features])
        scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_features)

        self.X = pd.concat([scaled_numerical_df, encoded_df], axis=1)
        self.y = self.label_encoder.fit_transform(self.df['Engagement_Level'])

    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def build_model(self):
        """Build the neural network model."""
        self.model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.label_encoder.classes_), activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train_model(self, epochs=20, batch_size=32, validation_split=0.2):
        """Train the model with early stopping."""
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )

    def evaluate_model(self):
        """Evaluate the model and store accuracy and classification report."""
        y_pred_probs = self.model.predict(self.X_test, verbose=0)
        y_pred = y_pred_probs.argmax(axis=1)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.classification_report = classification_report(
            self.y_test, y_pred, target_names=self.label_encoder.classes_, output_dict=True
        )
        return self.accuracy, self.classification_report

    def plot_training_curves(self, output_file='training_curves.png'):
        """Plot training and validation loss/accuracy curves."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.savefig(output_file)
        plt.close()

    def run_pipeline(self):
        """Run the entire pipeline."""
        self.load_and_process_data()
        self.encode_features()
        self.split_data()
        self.build_model()
        self.train_model()
        self.evaluate_model()
        self.plot_training_curves()