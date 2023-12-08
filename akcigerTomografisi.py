import numpy as np
import PIL.Image as img
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt

def get_file_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]

def convert_data(folder_path, class_label):
    images = get_file_paths(folder_path)
    data = []
    
    for image_path in images:
        image = img.open(image_path).convert("L")
        resized_image = image.resize((28, 28))
        flattened_image = np.array(resized_image).flatten()
        
        if class_label == 'covidli':
            data_point = np.append(flattened_image, [0])
        elif class_label == "covid_olmayan":
            data_point = np.append(flattened_image, [1])
        else:
            continue
        
        data.append(data_point)
    
    return data

covidli_folder = "C:/Users/USER/Desktop/COVID"
non_covid_folder = "C:/Users/USER/Desktop/non-COVID"

covidli_data = convert_data(covidli_folder, class_label="covidli")
covidli_df = pd.DataFrame(covidli_data)

non_covid_data = convert_data(non_covid_folder, class_label="covid_olmayan")
non_covid_df = pd.DataFrame(non_covid_data)

all_data = pd.concat([covidli_df, non_covid_df])

X = np.array(all_data)[:, :784]
y = np.array(all_data)[:, 784]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
clf = model.fit(X_train, y_train)

predictions = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)
roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.plot(fpr, tpr,'b', label='AUC = %0.2f' % roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend(loc='lower right')
plt.show()
