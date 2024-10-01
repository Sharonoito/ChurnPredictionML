import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


def clean_data(df):
    # Step 1: Handle missing data
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')  # Adjusted to total_charges
    df.fillna({'total_charges': df['total_charges'].mean()}, inplace=True)  # Adjusted to total_charges

    # Step 2: Encode categorical variables
    label_encoder = LabelEncoder()
    df['gender'] = label_encoder.fit_transform(df['gender'])  # Convert gender to numeric (0 or 1)

    # Other categorical fields to encode
    categorical_cols = ['multiple_lines', 'internet_service', 'online_security', 'online_backup', 
                        'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 
                        'contract', 'payment_method']
    
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col].astype(str))

    # Step 3: Normalize numerical features
    scaler = StandardScaler()
    df[['tenure', 'monthly_charges', 'total_charges']] = scaler.fit_transform(df[['tenure', 'monthly_charges', 'total_charges']])  # Adjusted to use total_charges

    return df


def split_data(df):
    # Ensure 'Churn' column exists
    if 'churn'not in df.columns:
        raise KeyError("'churn' column not found in the dataset")

    X = df.drop('churn', axis=1)  # Features
    y = df['churn'].astype(int)   # Target (convert to binary if needed)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)  # Increase max_iter for larger datasets
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Model Accuracy: {accuracy * 100:.2f}%")
    return model


def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")
    return model


def evaluate_model(y_test, y_pred):
    # Evaluate using confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
