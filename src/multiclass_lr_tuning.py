import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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
    # ========== 1. Load Dataset ==========
    dataset = load_dataset("AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset")
    data = dataset['train'].to_pandas()

    # ========== 2. Features and Labels ==========
    X = data["Short Description"].apply(custom_clean_text)
    y = data["Severity Label"]

    # ========== 3. Split Train/Test Set ==========
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    # ========== 4. Text Vectorization (TF-IDF) ==========
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=None,
        stop_words='english',
    )
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    # ========== 5. Train Logistic Regression (Without SMOTE) ==========
    clf_no_smote = LogisticRegression(
        C=30,
        class_weight='balanced',
        max_iter=270,
        solver='lbfgs',
        random_state=1
    )
    clf_no_smote.fit(X_train_vec, y_train)
    y_pred_no_smote = clf_no_smote.predict(X_test_vec)

    # ========== 6. Evaluation Metrics ==========
    acc_no_smote = accuracy_score(y_test, y_pred_no_smote)
    f1_macro_no_smote = f1_score(y_test, y_pred_no_smote, average='macro')
    print("=== Logistic Regression (No SMOTE) ===")
    print("Test Accuracy:", acc_no_smote)
    print("Test Macro F1-score:", f1_macro_no_smote)
    print("Classification Report:\n", classification_report(y_test, y_pred_no_smote))

    # ========== 7. Confusion Matrix Visualization ==========
    unique_labels = sorted(y.unique().tolist())
    conf_mat_no_smote = confusion_matrix(y_test, y_pred_no_smote, labels=unique_labels)

    plt.figure(figsize=(7, 5))
    sns.heatmap(conf_mat_no_smote, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Confusion Matrix (LogisticRegression, No SMOTE)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
