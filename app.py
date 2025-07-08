import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
import lime
import lime.lime_tabular
import webbrowser
import sys
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pickle
import json

# Thiết lập encoding cho stdout
sys.stdout.reconfigure(encoding='utf-8')

# Load dataset
df = pd.read_csv('Agriculture_dataset.csv', usecols=lambda col: "Unnamed" not in col)
Y = df["label"]
X = df[["N", "P", "K", "ph", "temperature", "humidity", "rainfall"]]

# Define base models
base_models = [
    ('dt', DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=0)),
    ('lr', LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
    ('nb', GaussianNB()),
    ('rf', RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)),
    ('svm', SVC(kernel='linear', random_state=42, probability=True))
]

# Define meta-learner
meta_learner = MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, random_state=42)

# Create stacking classifier
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=10,
    stack_method='auto'
)

# Stratified KFold cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store metrics
f1_scores = []
accuracy_scores = []
precision_scores = []
recall_scores = []

# Lưu X_test_scaled, X_test (chưa chuẩn hóa), và y_test để sử dụng sau
X_test_scaled_all = []
X_test_all = []
y_test_all = []

# Cross-validation loop
for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    # Feature Scaling using StandardScaler
    sc_X = StandardScaler()
    X_train_scaled = sc_X.fit_transform(X_train)
    X_test_scaled = sc_X.transform(X_test)
    
    # Lưu dữ liệu test để sử dụng sau
    X_test_scaled_all.append(X_test_scaled)
    X_test_all.append(X_test)
    y_test_all.append(y_test)
    
    # Fit stacking model
    stacking_model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = stacking_model.predict(X_test_scaled)
    
    # Calculate metrics
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, average='macro'))
    recall_scores.append(recall_score(y_test, y_pred, average='macro'))

# Print average performance metrics
print("Stacking Classifier Performance:")
print(f"Average F1-score: {np.mean(f1_scores):.4f}")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Average Precision: {np.mean(precision_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f}")

# Train final model on all data for prediction feature
final_scaler = StandardScaler()
X_scaled_final = final_scaler.fit_transform(X)
stacking_model.fit(X_scaled_final, Y)

# Save model and scaler for prediction
with open('stacking_model.pkl', 'wb') as f:
    pickle.dump(stacking_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(final_scaler, f)

# Get feature statistics for input validation
feature_stats = {
    'N': {'min': float(X['N'].min()), 'max': float(X['N'].max()), 'mean': float(X['N'].mean())},
    'P': {'min': float(X['P'].min()), 'max': float(X['P'].max()), 'mean': float(X['P'].mean())},
    'K': {'min': float(X['K'].min()), 'max': float(X['K'].max()), 'mean': float(X['K'].mean())},
    'ph': {'min': float(X['ph'].min()), 'max': float(X['ph'].max()), 'mean': float(X['ph'].mean())},
    'temperature': {'min': float(X['temperature'].min()), 'max': float(X['temperature'].max()), 'mean': float(X['temperature'].mean())},
    'humidity': {'min': float(X['humidity'].min()), 'max': float(X['humidity'].max()), 'mean': float(X['humidity'].mean())},
    'rainfall': {'min': float(X['rainfall'].min()), 'max': float(X['rainfall'].max()), 'mean': float(X['rainfall'].mean())}
}

# Get unique crop labels for prediction
crop_labels = sorted(Y.unique().tolist())

# Feature importance (dựa trên RandomForestClassifier trong base models)
rf_model = [model for name, model in base_models if name == 'rf'][0]
rf_model.fit(X_scaled_final, Y)  # Fit lại RandomForest để lấy feature importance
feature_names = ["Đạm (N)", "Lân (P)", "Kali (K)", "pH", "Nhiệt độ", "Độ ẩm", "Lượng mưa"]
feature_importance = rf_model.feature_importances_
feature_importance_dict = dict(zip(feature_names, feature_importance))

# Tạo biểu đồ tầm quan trọng đặc trưng
plt.figure(figsize=(10, 6))
plt.bar(feature_names, feature_importance, color='#4CAF50')
plt.title("Tầm quan trọng của đặc trưng (RandomForest)")
plt.xlabel("Đặc trưng")
plt.ylabel("Tầm quan trọng")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
buffer = BytesIO()
plt.savefig(buffer, format='png', bbox_inches='tight')
feature_importance_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Tạo biểu đồ hiệu suất mô hình
metrics = ['F1-score', 'Accuracy', 'Precision', 'Recall']
values = [np.mean(f1_scores), np.mean(accuracy_scores), np.mean(precision_scores), np.mean(recall_scores)]
plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color='#2196F3')
plt.title("Biểu đồ hiệu suất", fontsize=14, pad=20)
plt.ylabel("Giá trị", fontsize=12)
plt.ylim(0, 1)
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=10)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
buffer = BytesIO()
plt.savefig(buffer, format='png', bbox_inches='tight')
performance_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Tạo LIME explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_scaled_final,
    feature_names=feature_names,
    class_names=crop_labels,
    mode='classification'
)

# Giải thích một mẫu
i = 0
X_test_scaled = X_test_scaled_all[-1]
X_test = X_test_all[-1]
y_test = y_test_all[-1]
lime_exp = lime_explainer.explain_instance(
    data_row=X_test_scaled[i],
    predict_fn=stacking_model.predict_proba,
    num_features=7
)
predicted_crop = stacking_model.predict(X_test_scaled[i:i+1])[0]
true_crop = y_test.iloc[i]

# Lấy giá trị đặc trưng gốc của mẫu
sample_features = X_test.iloc[i]
sample_features_text = "\n".join([f"- {name}: {value:.4f}" for name, value in zip(feature_names, sample_features)])

# Lưu biểu đồ LIME dưới dạng base64
lime_exp.as_pyplot_figure()
plt.title(f"Giải thích LIME cho mẫu {i} (Dự đoán: {predicted_crop})")
plt.tight_layout()
buffer = BytesIO()
plt.savefig(buffer, format='png', bbox_inches='tight')
lime_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Tạo văn bản giải thích chi tiết bằng tiếng Việt
explanation_text = f"""
- **Cây trồng thực tế**: {true_crop}
- **Cây trồng dự đoán**: {predicted_crop}
- **Dự đoán đúng**: {'Có' if true_crop == predicted_crop else 'Không'}

**Giá trị đặc trưng của mẫu (trước chuẩn hóa):**
{sample_features_text}

**Tầm quan trọng của đặc trưng (Toàn cục, dựa trên RandomForest trong stacking):**
{chr(10).join([f'- {feature}: {importance:.4f}' for feature, importance in feature_importance_dict.items()])}

**Giải thích cục bộ từ LIME:**
Mô hình dự đoán '{predicted_crop}' dựa trên các yếu tố chính sau:
{chr(10).join([f'- {exp[0]}: {exp[1]:.4f}' for exp in lime_exp.as_list()])}
"""

# =============== FLASK API ===============
from flask import Flask, request, jsonify
from flask_cors import CORS

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

# Load model và scaler đã train
with open('stacking_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Danh sách crop labels
crop_labels = sorted(Y.unique().tolist())

@app.route('/')
def home():
    return "<h1>Crop Prediction API is running!</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Lấy input
        features = np.array([[
            float(data["N"]),
            float(data["P"]),
            float(data["K"]),
            float(data["ph"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["rainfall"])
        ]])

        # Chuẩn hóa
        features_scaled = scaler.transform(features)

        # Dự đoán
        pred_label = model.predict(features_scaled)[0]
        pred_probs = model.predict_proba(features_scaled)[0]

        # Trả JSON
        result = {
            "prediction": pred_label,
            "probabilities": {
                label: float(round(prob, 4)) 
                for label, prob in zip(crop_labels, pred_probs)
            }
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
