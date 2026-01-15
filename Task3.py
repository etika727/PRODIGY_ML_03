import os
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATASET_PATH = r"C:\Users\win 10\Downloads\train\train"
IMG_SIZE = 64
SAMPLES_PER_CLASS = 200   # 100 cats + 100 dogs

def extract_features(path):
    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    return hog(img, orientations=9,
               pixels_per_cell=(8,8),
               cells_per_block=(2,2))

X, y = [], []

cat_count = 0
dog_count = 0

print("Loading images...")

for file in os.listdir(DATASET_PATH):
    if not file.endswith(".jpg"):
        continue

    path = os.path.join(DATASET_PATH, file)

    try:
        if file.startswith("cat") and cat_count < SAMPLES_PER_CLASS:
            X.append(extract_features(path))
            y.append(0)
            cat_count += 1

        elif file.startswith("dog") and dog_count < SAMPLES_PER_CLASS:
            X.append(extract_features(path))
            y.append(1)
            dog_count += 1

        if cat_count == SAMPLES_PER_CLASS and dog_count == SAMPLES_PER_CLASS:
            break
    except:
        pass

print("Cats:", cat_count, "Dogs:", dog_count)

X = np.array(X)
y = np.array(y)

# -------- Train-Test Split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------- Train FAST SVM --------
print("Training model...")
model = LinearSVC(max_iter=3000)
model.fit(X_train, y_train)

# -------- Evaluation --------
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
