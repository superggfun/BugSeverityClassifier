import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import RandomOverSampler

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# ======= Text Cleaning Functions =======
def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in str(text).split() if word.lower() not in stop_words])

def full_clean_text(text):
    text = remove_html(text)
    text = remove_stopwords(text)
    return text

# ======= Load Bugzilla Dataset =======
dataset = load_dataset("AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset")
data = dataset['train'].to_pandas()

# Text Cleaning
print("Cleaning text...")
data["Processed_Description"] = data["Short Description"].apply(full_clean_text)
data["text_length"] = data["Processed_Description"].apply(lambda x: len(x.split()))
data = data[data["text_length"] >= 3]
print(f"Number of samples after cleaning: {len(data)}")

# Features and Labels
X = data["Processed_Description"]
y = data["Severity Label"]

# ======= Train/Test Split =======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======= TF-IDF Feature Extraction =======
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=None,
    token_pattern=r"(?u)\b\w{2,}\b"
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ======= Model without ROS =======
nb_no_ros = MultinomialNB(alpha=1.0)
nb_no_ros.fit(X_train_tfidf, y_train)
y_pred_no_ros = nb_no_ros.predict(X_test_tfidf)

acc_no_ros = accuracy_score(y_test, y_pred_no_ros)
f1_no_ros = f1_score(y_test, y_pred_no_ros, average='macro')
report_no_ros = classification_report(y_test, y_pred_no_ros)

print("\n=== Model without ROS ===")
print("Test Accuracy:", acc_no_ros)
print("Test Macro F1-score:", f1_no_ros)
print("Classification Report:\n", report_no_ros)

# Save accuracy results
results = [("No ROS", acc_no_ros)]

# ======= Train Models with ROS using Different random_state =======
ros_accuracies = []

for i in range(5):
    ros = RandomOverSampler(random_state=i)
    X_bal, y_bal = ros.fit_resample(X_train_tfidf, y_train)

    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_bal, y_bal)
    y_pred = nb.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    ros_accuracies.append(acc)

    # Output more metrics only when random_state=0
    if i == 0:
        f1_val = f1_score(y_test, y_pred, average='macro')
        report_val = classification_report(y_test, y_pred)
        print(f"\n=== Model with ROS (random_state={i}) ===")
        print("Test Accuracy:", acc)
        print("Test Macro F1-score:", f1_val)
        print("Classification Report:\n", report_val)

    results.append((f"ROS random_state={i}", acc))

    if i == 0:
        # ======= Plot Confusion Matrices (on separate plots) =======
        unique_labels = sorted(y.unique().tolist())
        conf_mat_no_ros = confusion_matrix(y_test, y_pred_no_ros, labels=unique_labels)
        conf_mat_ros = confusion_matrix(y_test, y_pred, labels=unique_labels)

        # Confusion matrix without ROS
        plt.figure(figsize=(7, 5))
        sns.heatmap(conf_mat_no_ros, annot=True, fmt='d', cmap='Blues',
                    xticklabels=unique_labels, yticklabels=unique_labels)
        plt.title('Confusion Matrix (No ROS)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

        # Confusion matrix with ROS
        plt.figure(figsize=(7, 5))
        sns.heatmap(conf_mat_ros, annot=True, fmt='d', cmap='Blues',
                    xticklabels=unique_labels, yticklabels=unique_labels)
        plt.title('Confusion Matrix (ROS random_state=0)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()

# ======= Calculate Mean and Standard Deviation of Accuracies =======
mean_acc = sum(ros_accuracies) / len(ros_accuracies)
std_acc = (sum((x - mean_acc) ** 2 for x in ros_accuracies) / len(ros_accuracies)) ** 0.5
results.append(("ROS Mean", mean_acc))
results.append(("ROS Std", std_acc))

# ======= Print Accuracy Comparison Table =======
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
print("\nAccuracy Comparison:")
print(results_df)
