import os
import io
import pickle
import joblib
import numpy as np
import pandas as pd
import pymysql
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request, redirect, url_for, flash, session

app = Flask(__name__)
app.secret_key = "a_very_secret_key"

MODEL_DIR = "model"
DATASET_DIR = "Dataset"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

MYSQL_HOST = 'localhost'
MYSQL_USER = 'root'
MYSQL_PASSWORD = 'root'
MYSQL_DB = 'har_db'

def get_db_connection():
    try:
        return pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            db=MYSQL_DB,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
    except Exception as e:
        print(f"DB connection failed: {e}")
        return None


def check_user_credentials(username, password):
    conn = get_db_connection()
    if not conn:
        return False, None
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT password, role FROM users WHERE username=%s", (username,))
            record = cursor.fetchone()
            if record and record["password"] == password:
                return True, record["role"]
            return False, None
    except Exception as e:
        print(f"Error verifying credentials: {e}")
        return False, None
    finally:
        conn.close()


def register_new_user(username, password, email):
    conn = get_db_connection()
    if not conn:
        return False, "Database connection failed."
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (username, password, email, role) VALUES (%s, %s, %s, 'user')",
                (username, password, email)
            )
        conn.commit()
        return True, "User registered successfully."
    except pymysql.err.IntegrityError as e:
        if "Duplicate entry" in str(e):
            if "users.username" in str(e):
                return False, "Username already exists."
            elif "users.email" in str(e):
                return False, "Email already exists."
        return False, "Integrity error while registering user."
    except Exception as e:
        return False, f"Unexpected error: {e}"
    finally:
        conn.close()

print("[INFO] Loading ELECTRA model and tokenizer...")
ELECTRA_MODEL_NAME = "google/electra-base-discriminator"
tokenizer = AutoTokenizer.from_pretrained(ELECTRA_MODEL_NAME)
model = AutoModel.from_pretrained(ELECTRA_MODEL_NAME)
model.eval()
print("[INFO] ELECTRA loaded.")


def preprocess_data(df, target_cols=None, is_train=False):
    global label_encoders
    label_encoders = {}
    df = df.copy()
    df = df.dropna(how="all").reset_index(drop=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def clean_text(text):
        text = "" if pd.isna(text) else str(text).strip().lower()
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words]
        return " ".join(tokens)

    target_df = None
    if target_cols:
        existing_targets = [c for c in target_cols if c in df.columns]
        if existing_targets:
            target_df = df[existing_targets].copy()
            df = df.drop(columns=existing_targets)

    text_columns = df.select_dtypes(include="object").columns.tolist()
    text_columns = sorted(text_columns)

    for col in text_columns:
        df[f"processed_{col}"] = df[col].apply(clean_text)

    if text_columns:
        df = df.drop(columns=text_columns)

    if target_df is not None:
        for col in target_df.columns:
            df[col] = target_df[col]

    processed_text_cols = [c for c in df.columns if c.startswith("processed_")]
    processed_text_cols = sorted(processed_text_cols)
    X_text = df[processed_text_cols].astype(str).agg(" ".join, axis=1).tolist()

    if is_train and target_cols:
        Y_dict = {}
        for col in target_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = df[col].astype(str).str.strip()
                Y_dict[col] = le.fit_transform(df[col])
                label_encoders[col] = le
        return X_text, Y_dict, label_encoders
    else:
        return X_text, {}


def electra_feature_extraction(texts, batch_size=32, pooling="mean"):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting ELECTRA embeddings"):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = model(**encoded)
        token_embeddings = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        if pooling == "mean":
            sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
            sum_mask = mask.sum(dim=1)
            embeddings = sum_embeddings / sum_mask
        else:
            embeddings = token_embeddings[:, 0, :]
        all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        valid, role = check_user_credentials(username, password)
        if valid:
            session["username"] = username
            session["role"] = role
            flash(f"Welcome, {username}!", "success")
            return redirect(url_for("predict"))
        flash("Invalid username or password.", "danger")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        email = request.form.get("email")
        success, msg = register_new_user(username, password, email)
        if success:
            flash(msg, "success")
            return redirect(url_for("login"))
        else:
            flash(msg, "danger")
    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("home"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "username" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("login"))

    if request.method == "POST":
        uploaded_file = request.files.get("dataset")
        if not uploaded_file or uploaded_file.filename == "":
            flash("Please upload a CSV file!", "danger")
            return redirect(url_for("predict"))

        # Save uploaded file temporarily
        file_path = os.path.join(DATASET_DIR, uploaded_file.filename)
        uploaded_file.save(file_path)
        flash(f"Dataset '{uploaded_file.filename}' uploaded successfully. Running predictions...", "info")

        try:
            # Load training dataset for label encoders
            train_data_path = os.path.join(DATASET_DIR, "mtsamples.csv")
            df_train = pd.read_csv(train_data_path)
            if "Unnamed: 0" in df_train.columns:
                df_train = df_train.drop(columns=["Unnamed: 0"])
            _, _, label_encoders = preprocess_data(df_train, target_cols=["medical_specialty"], is_train=True)

            # Load test dataset (uploaded)
            df_test = pd.read_csv(file_path)
            X_test, _ = preprocess_data(df_test, is_train=False)
            features_test = electra_feature_extraction(X_test)

            # Load model
            model_path = os.path.join(MODEL_DIR, "ELECTRA_word_embeddings_medical_specialty_ETC_model.npz")
            npzfile = np.load(model_path, allow_pickle=True)
            mdl = pickle.loads(npzfile["model"].item())

            # Predict
            le = label_encoders["medical_specialty"]
            y_pred = mdl.predict(features_test)
            mapped_labels = le.inverse_transform(y_pred)
            df_test["Predicted_medical_specialty"] = mapped_labels

            flash("Prediction completed successfully!", "success")

            # Display top 20 results
            return render_template(
                "predict_result.html",
                tables=[df_test.head(20).to_html(classes="table table-striped table-bordered", index=False)],
                dataset_name=uploaded_file.filename
            )

        except Exception as e:
            flash(f"Error during prediction: {e}", "danger")
            return redirect(url_for("predict"))

    return render_template("predict.html")



if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
