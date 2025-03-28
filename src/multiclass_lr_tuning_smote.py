import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import re

def custom_clean_text(text: str) -> str:
    """
    Custom text cleaning function:
    1) Replace underscores '_' with spaces ' '.
    2) Filter out tokens with length <= 3 that contain digits (e.g., 'v1', '1.2', etc.).
    3) Keep all other tokens.
    """
    tokens = text.split()
    cleaned_tokens = []
    for token in tokens:
        token_no_underscore = token.replace('_', ' ')
        if len(token_no_underscore) <= 3 and re.search(r'\d', token_no_underscore):
            continue
        cleaned_tokens.append(token_no_underscore.strip())
    return " ".join(cleaned_tokens)

def main():
    # 1. Load dataset
    dataset = load_dataset("AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset")
    data = dataset['train'].to_pandas()

    # 2. Features and labels
    X = data["Short Description"].apply(custom_clean_text)
    y = data["Severity Label"]

    # 3. Split into train/test sets
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    # 4. Text vectorization
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=None,
        stop_words='english',
    )
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    # 5. Apply SMOTE for oversampling
    smote = SMOTE(random_state=1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)

    # 6. Define logistic regression classifier
    clf = LogisticRegression(
        C=30,
        class_weight='balanced',
        max_iter=270,
        solver='lbfgs',
        random_state=1
    )

    # 7. Train model
    clf.fit(X_train_resampled, y_train_resampled)

    # 8. Evaluate on the test set
    y_pred = clf.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro_value = f1_score(y_test, y_pred, average='macro')
    print("\nTest Accuracy:", accuracy)
    print("Test Macro F1-score:", f1_macro_value)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # 9. Confusion matrix visualization
    unique_labels = sorted(y.unique().tolist())
    conf_mat = confusion_matrix(y_test, y_pred, labels=unique_labels)
    plt.figure(figsize=(7, 5))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Confusion Matrix (SMOTE + LogisticRegression)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

if __name__ == "__main__":
    main()
