import glob
from PIL import Image
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
import joblib


smile_images=glob.glob('data/smile/*.jpg')
non_smile_images=glob.glob('data/non_smile/*.jpg')

data=[]
labels=[]

for image_path in smile_images:
    image=Image.open(image_path).convert('L')
    image=image.resize((64,64))
    image=np.array(image).flatten()

    data.append(image)
    labels.append(1)

for image_path in non_smile_images:
    image=Image.open(image_path).convert('L')
    image=image.resize((64,64))
    image=np.array(image).flatten()

    data.append(image)
    labels.append(0)

x=np.array(data)
y=np.array(labels)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)



y_pred=model.predict(x_test)


accuracy = accuracy_score(y_test, y_pred)
cm=confusion_matrix(y_test, y_pred)
report=classification_report(y_test, y_pred, output_dict=True)


metrics = {
    "accuracy": accuracy,
    "confusion_matrix": cm,
    "classification_report": report
}






joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(metrics, "metrics.pkl")

print("Model Saved Successfully")



