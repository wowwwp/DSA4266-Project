import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('Features_For_Traditional_ML_Techniques.csv')

# Extract the tweet text and labels
tweets = df['tweet'].astype(str).values  # Using 'tweet' as the text input
labels = df['majority_target'].values  # Assuming this is the target label

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_features = vectorizer.fit_transform(tweets)  # Keep this as a sparse matrix

# Combine TF-IDF features with the rest of your features
extra_features = df.drop(columns=['tweet', 'statement', 'majority_target', 'BinaryNumTarget'])  # Drop non-feature columns

# Ensure all extra features are numeric, you may need to convert or fill NaNs
extra_features = extra_features.apply(pd.to_numeric, errors='coerce')  # Convert to numeric, coercing errors to NaN
extra_features.fillna(0, inplace=True)  # Fill NaN values with 0 (or another strategy)

# Combine the features (TF-IDF sparse matrix with the dense feature matrix)
from scipy.sparse import hstack
combined_features = hstack([extra_features, tfidf_features])  # Keep this combined as a sparse matrix

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.3, random_state=42)

# Create and train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
