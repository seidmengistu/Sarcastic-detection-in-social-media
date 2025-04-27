# from xgboost import XGBClassifier
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Classical ML models
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import LinearSVC
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# from sklearn.metrics import classification_report, accuracy_score
# from ..utils.preprocessing import preprocess_text
# import os
# # ================ LOAD & PREPROCESS DATA ================
# # Replace with your actual file path
# PROJECT_ROOT_DIR = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "../."))
# # Paths to data directories
# DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
# PROCESSED_DATA_DIR = os.path.join(
#     DATA_DIR, "processed", 'dataset_unificato.csv')
# df = pd.read_csv(PROCESSED_DATA_DIR)

# # Assume columns: "text" and "sarcastic"
# df["text"] = df["text"].dropna()
# df['class'].replace({"notsarc": 0, "sarc": 1}, inplace=True)
# df["processed_text"] = df["text"].apply(lambda x: preprocess_text(x))

# # ================ SPLIT ===================
# X_train, X_test, y_train, y_test = train_test_split(
#     df["processed_text"],
#     df["class"],
#     test_size=0.3,
#     shuffle=True,
#     random_state=42,
#     stratify=df["class"]
# )

# # ================ TF-IDF VECTORIZER ================
# vectorizer = TfidfVectorizer(
#     max_features=10000, ngram_range=(1, 2))
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)


# # ================ MODELS ==========================
# models = {
#     "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
#     "Naive Bayes": MultinomialNB(),
#     "SVM": LinearSVC(class_weight='balanced'),
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#     "XGBoost": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss'),
#     "Ensemble (Voting)": VotingClassifier(
#         estimators=[
#             ('lr', LogisticRegression(max_iter=1000)),
#             ('rf', RandomForestClassifier()),
#             ('svm', LinearSVC())
#         ],
#         voting='hard'
#     )
# }

# # ================ TRAIN & EVALUATE =====================
# for name, model in models.items():
#     print(f"\n=== {name} ===")
#     model.fit(X_train_tfidf, y_train)
#     y_pred = model.predict(X_test_tfidf)
#     acc = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {acc:.4f}")
#     print(classification_report(y_test, y_pred,
#           target_names=["Not Sarcastic", "Sarcastic"]))
import os
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from ..utils.preprocessing import preprocess_text  # Adjust path if needed

# ================ LOAD & PREPROCESS DATA ================
PROJECT_ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../."))
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
PROCESSED_DATA_PATH = os.path.join(
    DATA_DIR, "processed", "dataset_unificato.csv")

df = pd.read_csv(PROCESSED_DATA_PATH)

df["text"] = df["text"].dropna()
df['class'].replace({"notsarc": 0, "sarc": 1}, inplace=True)
df["processed_text"] = df["text"].apply(lambda x: preprocess_text(x))

# ================ SPLIT INTO train / val / test ===================
X_temp, X_test, y_temp, y_test = train_test_split(
    df["processed_text"], df["class"],
    test_size=0.2, stratify=df["class"], random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25, stratify=y_temp, random_state=42  # 0.25 * 0.8 = 0.2 total
)

# ================ MODELS & PARAM GRIDS ====================
models_and_params = {
    "Logistic Regression": {
        'model': LogisticRegression(class_weight='balanced', max_iter=1000),
        'params': {
            'clf__C': [0.01, 0.1, 1, 10]
        }
    },
    "SVM": {
        'model': LinearSVC(class_weight='balanced'),
        'params': {
            'clf__C': [0.1, 1, 10]
        }
    },
    "Random Forest": {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [None, 10, 20]
        }
    },
    "XGBoost": {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'params': {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [3, 6]
        }
    }
}

# ================ TRAIN, TUNE & EVALUATE ====================
for name, mp in models_and_params.items():
    print(f"\n🔍 Tuning {name}")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('clf', mp['model'])
    ])

    grid = GridSearchCV(
        pipeline,
        param_grid=mp['params'],
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print(f"\n✅ Best Parameters for {name}: {grid.best_params_}")

    # ============ VALIDATION EVALUATION ============
    val_preds = grid.predict(X_val)
    print(f"\n📊 Validation Report for {name}:")
    print(classification_report(y_val, val_preds,
          target_names=["Not Sarcastic", "Sarcastic"]))

    # ============ TEST EVALUATION ============
    test_preds = grid.predict(X_test)
    print(f"\n📈 Final Test Report for {name}:")
    print(classification_report(y_test, test_preds,
          target_names=["Not Sarcastic", "Sarcastic"]))
