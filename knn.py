from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
iris = datasets.load_iris()
print("Iris dataset is loaded")
print("Dataset is split into training and testing sets")
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, 
print("Training data is: ", x_train.shape, y_train.shape)
print("Testing data is: ", x_test.shape, y_test.shape)
for i in range(len(iris.target_names)):
 print("Label ", i, " - ", str(iris.target_names[i]))
 
k = 5
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print("Results of classification is\n")
for i in range(0,len(x_test)):
 print("Sample : ", str(x_test[i]), " Actual Label: ", str(y_test[i]), " 
 "Accuracy : ", classifier.score(x_test, y_test))
 
from sklearn.metrics import classification_report, confusion_matrix
print(" Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
print(" Accuracy Matrix: \n", classification_report(y_test, y_pred))
