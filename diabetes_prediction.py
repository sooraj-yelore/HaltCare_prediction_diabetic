import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings

warnings.filterwarnings('ignore')


# Load dataset
def load_dataset():
    # Using the Pima Indians Diabetes Dataset
    # You can replace this with your actual dataset
    url = "https://raw.githubusercontent.com/Simplilearn-Edu/Data-Science-Capstone-Projects/refs/heads/master/health%20care%20diabetes.csv"
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

    try:
        # Try to load from URL
        df = pd.read_csv(url, header=None, names=column_names)
        print("✓ Dataset loaded from URL")
    except:
        # If offline, try to load local file
        try:
            df = pd.read_csv('diabetes.csv')
            print("✓ Dataset loaded from local file")
        except:
            print("Error: Could not load dataset")
            print("Please download the dataset from:")
            print("https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
            return None

    return df


def train_model():
    print("🚀 Starting model training...")

    # Load data
    df = load_dataset()
    if df is None:
        return False

    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()[:-1]}")

    # Handle missing/zero values (common in this dataset)
    # For glucose, blood pressure, etc., zero values are biologically impossible
    columns_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in columns_to_clean:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    # Split features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    print(f"\nClass distribution:")
    print(f"Non-diabetic: {(y == 0).sum()} ({((y == 0).sum() / len(y) * 100):.1f}%)")
    print(f"Diabetic: {(y == 1).sum()} ({((y == 1).sum() / len(y) * 100):.1f}%)")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM model
    print("\n🔧 Training SVM model...")
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n📊 Model Performance:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model and scaler
    print("\n💾 Saving model files...")
    joblib.dump(model, 'diabetes_model.pkl')
    joblib.dump(scaler, 'scaler_svm.pkl')

    print("✅ Model saved as 'diabetes_model.pkl'")
    print("✅ Scaler saved as 'scaler_svm.pkl'")

    # Feature importance (for linear kernel)
    if hasattr(model, 'coef_'):
        print(f"\n📈 Feature Importance:")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': np.abs(model.coef_[0])
        }).sort_values('Importance', ascending=False)
        print(feature_importance)

    return True


if __name__ == "__main__":
    success = train_model()
    if success:
        print("\n🎉 Training complete! You can now run:")
        print("streamlit run app.py")
    else:
        print("\n❌ Training failed!")