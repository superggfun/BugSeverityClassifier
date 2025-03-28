import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class TwoStageBugClassifier:
    """
    两阶段分类器：
    1) Phase 1: Binary Classification (normal vs not_normal)
    2) Phase 2: Multi-class Classification (blocker, critical, major, minor, trivial)
    """

    def __init__(
        self,
        ngram_range_bin=(1, 2),
        max_features_bin=None,
        ngram_range_multi=(1, 2),
        max_features_multi=None,
        stop_words="english",
        token_pattern=r"(?u)\b\w{2,}\b",
        random_state=1,
        C=30,
        max_iter=280,
        class_weight="balanced",
    ):
        """
        构造函数：初始化向量化器与分类器等组件。
        """
        # 阶段1（binary）
        self.tfidf_bin = TfidfVectorizer(
            ngram_range=ngram_range_bin,
            max_features=max_features_bin,
            stop_words=stop_words,
            token_pattern=token_pattern,
        )
        self.clf_bin = LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            C=C,
            random_state=random_state,
        )

        # 阶段2（multi-class）
        self.tfidf_multi = TfidfVectorizer(
            ngram_range=ngram_range_multi,
            max_features=max_features_multi,
            stop_words=stop_words,
            token_pattern=token_pattern,
        )
        self.clf_multi = LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            C=C,
            random_state=random_state,
        )

    def fit_phase1(self, X_train, y_train):
        """
        训练阶段1：Binary Classification (normal vs not_normal)
        """
        X_train_tfidf = self.tfidf_bin.fit_transform(X_train)
        self.clf_bin.fit(X_train_tfidf, y_train)

    def evaluate_phase1(self, X_test, y_test):
        """
        测试阶段1模型，并打印结果
        """
        X_test_tfidf = self.tfidf_bin.transform(X_test)
        y_pred = self.clf_bin.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        print("[Phase 1] Accuracy (normal vs not_normal):", acc)
        print("[Phase 1] Classification Report:\n", classification_report(y_test, y_pred))

    def fit_phase2(self, X_train, y_train):
        """
        训练阶段2：对 not_normal 样本进行多分类
        """
        X_train_tfidf = self.tfidf_multi.fit_transform(X_train)
        self.clf_multi.fit(X_train_tfidf, y_train)

    def evaluate_phase2(self, X_test, y_test):
        """
        测试阶段2模型，并打印结果
        """
        X_test_tfidf = self.tfidf_multi.transform(X_test)
        y_pred = self.clf_multi.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        print("[Phase 2] Accuracy (Multi-class on not_normal):", acc)
        print("[Phase 2] Classification Report:\n", classification_report(y_test, y_pred))

    def predict_final(self, X_list):
        """
        两阶段推理：先用阶段1判断是否 normal，再用阶段2对 not_normal 做多分类。
        参数 X_list 为文本列表（或 Series），函数返回对应的标签列表。
        """
        # 先用Phase 1判断 normal / not_normal
        X_bin_tfidf = self.tfidf_bin.transform(X_list)
        y_pred_bin = self.clf_bin.predict(X_bin_tfidf)

        # 对每个样本判断：若为 normal，直接输出；否则进入多分类
        final_preds = []
        for i, text in enumerate(X_list):
            if y_pred_bin[i] == "normal":
                final_preds.append("normal")
            else:
                # Phase 2多分类
                x_tfidf = self.tfidf_multi.transform([text])
                pred_label = self.clf_multi.predict(x_tfidf)[0]
                final_preds.append(pred_label)
        return final_preds

    def evaluate_final(self, X_test, y_true, class_list=None, plot_matrix=True):
        """
        使用两阶段推理对 X_test 做预测，并与 y_true 对比，打印结果。
        可选地绘制混淆矩阵（需要传入 class_list）。
        """
        final_preds = self.predict_final(X_test)
        acc = accuracy_score(y_true, final_preds)
        print("\n[Combined] Two-Stage Model Accuracy:", acc)
        print("[Combined] Classification Report:\n", classification_report(y_true, final_preds))

        if plot_matrix and class_list is not None:
            conf_mat = confusion_matrix(y_true, final_preds, labels=class_list)
            plt.figure(figsize=(10, 7))
            sns.heatmap(
                conf_mat, annot=True, fmt='d',
                cmap='Blues', xticklabels=class_list, yticklabels=class_list
            )
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix - Two-Stage Model')
            plt.show()


def main():
    # ========== 1. 读取原始数据（完整） ==========
    dataset = load_dataset("AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset")
    data = dataset['train'].to_pandas()

    # ========== 2. 首先拿出最终测试集，避免数据泄露 ==========
    # 先把整份 data 拆分成 train_data & final_test_data
    train_data, final_test_data = train_test_split(data, test_size=0.2, random_state=1)

    # ========== 3. 在 train_data 上做两阶段训练 & 验证 ==========
    # ---------- 阶段1：normal vs not_normal ----------
    # 给训练区加一个 binary_label
    train_data['binary_label'] = train_data['Severity Label'].apply(
        lambda x: 'normal' if x == 'normal' else 'not_normal'
    )

    # 拿出二分类所需的 X, y
    X_bin = train_data["Short Description"]
    y_bin = train_data["binary_label"]

    # 在训练区内部，再划分一部分做验证（也可以不用拆分，看你需求）
    X_train_bin, X_val_bin, y_train_bin, y_val_bin = train_test_split(
        X_bin, y_bin, test_size=0.2, random_state=1
    )

    # ---------- 阶段2：multi-class（对 not_normal 的数据） ----------
    train_data_not_normal = train_data[train_data["Severity Label"] != "normal"]
    X_multi = train_data_not_normal["Short Description"]
    y_multi = train_data_not_normal["Severity Label"]

    X_train_multi, X_val_multi, y_train_multi, y_val_multi = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=1
    )

    # ========== 4. 初始化 & 训练两阶段模型 ==========
    model = TwoStageBugClassifier(
        ngram_range_bin=(1, 2),
        max_features_bin=80000,
        ngram_range_multi=(1, 2),
        max_features_multi=80000,
        stop_words="english",
        token_pattern=r"(?u)\b\w{2,}\b",
        random_state=1,
        C=10,
        max_iter=500,
        class_weight='balanced'
    )

    # 训练阶段1，并在验证集上评价
    model.fit_phase1(X_train_bin, y_train_bin)
    model.evaluate_phase1(X_val_bin, y_val_bin)

    # 训练阶段2，并在验证集上评价
    model.fit_phase2(X_train_multi, y_train_multi)
    model.evaluate_phase2(X_val_multi, y_val_multi)

    # ========== 5. 最终评估：对之前拆分出来的 final_test_data 做完整两阶段推理 ==========
    X_final_test = final_test_data["Short Description"]
    y_final_test = final_test_data["Severity Label"]

    # 混淆矩阵的类别列表：用「训练集中出现的 not_normal 类别 + 'normal'」
    unique_labels_phase2 = sorted(train_data_not_normal["Severity Label"].unique().tolist())
    class_list = unique_labels_phase2 + ["normal"]

    # 最终评估
    model.evaluate_final(
        X_final_test,
        y_final_test,
        class_list=class_list,
        plot_matrix=True
    )


if __name__ == "__main__":
    main()
