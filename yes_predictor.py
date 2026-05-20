import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# -----------------------------------
# 1. LOAD DATA
# -----------------------------------

df = pd.read_csv("data.csv")

# -----------------------------------
# 2. FEATURE ENGINEERING
# -----------------------------------

# Create flags for missing values
df['has_college'] = df['collegeName'].notnull().astype(int)
df['has_company'] = df['companyName'].notnull().astype(int)

# -----------------------------------
# 3. FEATURES AND TARGET
# -----------------------------------

features = [
    'groupSize',
    'category',
    'registerType',
    'state',
    'city',
    'has_college',
    'has_company'
]

X = df[features]
y = df['isPresent']

# -----------------------------------
# 4. TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# 5. PREPROCESSING PIPELINE
# -----------------------------------

preprocessor = ColumnTransformer([
    
    # Small categories
    ('low_card',
     OneHotEncoder(handle_unknown='ignore'),
     ['category', 'registerType']),
    
    # Large categories
    ('high_card',
     OrdinalEncoder(
         handle_unknown='use_encoded_value',
         unknown_value=-1
     ),
     ['state', 'city']),
    
    # Numerical features
    ('num',
     StandardScaler(),
     ['groupSize', 'has_college', 'has_company'])
])

# -----------------------------------
# 6. MODEL
# -----------------------------------

model = Pipeline([
    ('prep', preprocessor),
    
    ('clf', GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=5,
        random_state=42
    ))
])

# -----------------------------------
# 7. HANDLE CLASS IMBALANCE
# -----------------------------------

sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train
)

# -----------------------------------
# 8. TRAIN MODEL
# -----------------------------------

model.fit(
    X_train,
    y_train,
    clf__sample_weight=sample_weights
)

# -----------------------------------
# 9. PREDICTIONS
# -----------------------------------

y_pred = model.predict(X_test)

# -----------------------------------
# 10. OVERALL ACCURACY
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

print(f"\nOverall Accuracy: {accuracy:.2%}")

# -----------------------------------
# 11. YES / NO ACCURACY
# -----------------------------------

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

# YES accuracy
yes_accuracy = TP / (TP + FN)

# NO accuracy
no_accuracy = TN / (TN + FP)

print(f"YES Prediction Accuracy : {yes_accuracy:.2%}")
print(f"NO Prediction Accuracy  : {no_accuracy:.2%}")

# -----------------------------------
# 12. CONFUSION MATRIX DISPLAY
# -----------------------------------

print("\nConfusion Matrix:")
print(cm)