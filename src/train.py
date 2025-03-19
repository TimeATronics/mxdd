import cv2
import os
import numpy as np
import joblib
from tqdm import tqdm
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from tensorflow.keras.utils import plot_model # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = "mnet_xgboost1.pkl"
SCALER_PATH = "scaler_norm1.pkl"
FEATURES_PATH = "features_X1.npy"
LABELS_PATH = "features_y1.npy"

mobilenet = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

def extract_deep_features(image):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0) # Add batch dimension
    features = mobilenet.predict(image)
    return features.flatten()

def process_image(args):
    image_path, label = args
    image = cv2.imread(image_path)
    if image is None:
        return None, None    
    feature_vector = extract_deep_features(image)
    return feature_vector, label

def load_dataset(real_folder, fake_folder):
    if os.path.exists(FEATURES_PATH) and os.path.exists(LABELS_PATH):
        print("Loading precomputed features...")
        X = np.load(FEATURES_PATH)
        y = np.load(LABELS_PATH)
        print("Features loaded successfully!")
        return X, y
    X, y = [], []
    image_paths = [(os.path.join(real_folder, img), 0) for img in os.listdir(real_folder)] + \
                  [(os.path.join(fake_folder, img), 1) for img in os.listdir(fake_folder)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_image, image_paths), total=len(image_paths), desc="Extracting Features"))
    for feature_vector, label in results:
        if feature_vector is not None:
            X.append(feature_vector)
            y.append(label)
    X, y = np.array(X), np.array(y)
    np.save(FEATURES_PATH, X)
    np.save(LABELS_PATH, y)
    print("Features saved for future use.")
    return X, y

if __name__ == "__main__":
    real_folder = "C:\\Users\\AradhyaPC\\Desktop\\deepfake_detection\\real"
    fake_folder = "C:\\Users\\AradhyaPC\\Desktop\\deepfake_detection\\fake"
    print("\nExtracting deep features from images...")
    X, y = load_dataset(real_folder, fake_folder)
    print("Feature extraction complete.")
    print("\nSplitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1234)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1234)
    print("Dataset split complete!")
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("Loading saved model and scaler...")
        clf = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model loaded successfully!")
    else:
        print("\nTraining XGBoost model...")
        # Train XGBoost with Validation Set
        clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, eval_metric="logloss")
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

        # Save the model and scaler
        joblib.dump(clf, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        print(f"Model saved as {MODEL_PATH} & scaler saved as {SCALER_PATH}")
        #clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, eval_metric="logloss")
        #clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
        #print("Model training complete!")
        #joblib.dump(clf, MODEL_PATH)
        #joblib.dump(scaler, SCALER_PATH)

    # Evaluate Model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

    # Plot Validation Loss
    epochs = 300
    x_axis = range(0, epochs)
    results = clf.evals_result()
    plt.plot(x_axis, results["validation_0"]["logloss"], label="Validation Log Loss")
    plt.xlabel("Epochs", fontsize=16, fontweight="bold")
    plt.ylabel("Log Loss", fontsize=16, fontweight="bold")
    plt.legend()
    plt.title("Validation Log Loss over Epochs", fontsize=20, fontweight="bold")
    plt.show()

    #print("\nEvaluating model...")
    #y_pred = clf.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    #print(f"Model Accuracy: {accuracy * 100:.2f}%")

#    print("\nMobileNetV2 Model Summary:")
#    mobilenet.summary()
#    plot_model(mobilenet, to_file="mobilenetv2_structure.png", show_shapes=True, show_layer_names=True)
#    print("MobileNetV2 model structure saved as 'mobilenetv2_structure.png'")
#    print("\nXGBoost Model Summary:")
#    print(clf.get_booster().get_dump()[0])

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"], annot_kws={"size": 40})
    plt.xlabel("Predicted Label", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=16, fontweight="bold")
    plt.title("Confusion Matrix", fontsize=20, fontweight="bold")
    plt.savefig("confusion_matrix.png")
    plt.show()

    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=["Real", "Fake"])
    print(report)

    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate", fontsize=16, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=16, fontweight="bold")
    plt.title("ROC Curve for Deepfake Detection", fontsize=20, fontweight="bold")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.show()
    print(f"ROC Curve plotted. AUC Score: {roc_auc:.4f}")