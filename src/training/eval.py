import os, json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import itertools

ROOT = r"C:\Users\HP\Desktop\foodtask\food-logger"
MODEL_PATH = os.path.join(ROOT, "models", "food_model.h5")
LABELS_PATH = os.path.join(ROOT, "models", "labels.json")
TEST_DIR = os.path.join(ROOT, "data", "test")
IMG_SIZE = (224,224)

print("Loading model...")
model = load_model(MODEL_PATH)
with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    labels_map = json.load(f)   # expects {"0":"aloo_gobi", ...}
name_to_idx = {v:int(k) for k,v in labels_map.items()}
idx_to_name = {int(k):v for k,v in labels_map.items()}
print("Classes:", len(name_to_idx))

y_true = []
y_pred = []
filelist = []

for cls_name, idx in name_to_idx.items():
    cls_dir = os.path.join(TEST_DIR, cls_name)
    if not os.path.isdir(cls_dir):
        continue
    for fname in os.listdir(cls_dir):
        if not fname.lower().endswith(('.jpg','.jpeg','.png')):
            continue
        path = os.path.join(cls_dir, fname)
        try:
            img = load_img(path, target_size=IMG_SIZE)
            arr = img_to_array(img)/255.0
            arr = np.expand_dims(arr,0)
            preds = model.predict(arr)[0]
            pred_idx = int(np.argmax(preds))
            y_true.append(idx)
            y_pred.append(pred_idx)
            filelist.append(path)
        except Exception as e:
            print("Error reading", path, e)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
acc = accuracy_score(y_true, y_pred)
print("Test accuracy (top-1):", acc)

print("\nClassification report (per-class):")
target_names = [idx_to_name[i] for i in sorted(idx_to_name.keys())]
print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=sorted(idx_to_name.keys()))
np.save(os.path.join(ROOT, "models", "confusion_matrix.npy"), cm)
print("Saved confusion matrix to models/confusion_matrix.npy")
