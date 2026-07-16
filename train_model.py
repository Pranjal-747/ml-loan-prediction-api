import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

df = pd.read_csv("loan_data.csv")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop("Loan_Status")   # ← changed
categorical_cols = df.select_dtypes(include=["object"]).columns


# Handle missing values
num_imputer = SimpleImputer(strategy="median")
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# One-hot encode
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop("Loan_Status", axis=1)    # ← changed
y = df["Loan_Status"]                  # ← changed

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

print(f"Test Accuracy: {model.score(X_test, y_test):.3f}")

# Save the trained model AND the column names (critical — explained below)
joblib.dump(model, "loan_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("Model saved as loan_model.pkl")