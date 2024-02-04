import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read the CSV file
data = pd.read_csv('adult.csv')

# Step 2: Data Preprocessing
data = pd.get_dummies(data, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country'])
X = data.drop(columns=['income'])
y = data['income']

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the KNN model
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Step 5: Make Predictions with KNN
y_pred_knn = knn.predict(X_test)

# Calculate and store the accuracy of KNN
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# Step 6: Train the Weighted KNN model
knn_weighted = KNeighborsClassifier(n_neighbors=k, weights='distance')
knn_weighted.fit(X_train, y_train)

# Step 7: Make Predictions with Weighted KNN
y_pred_weighted = knn_weighted.predict(X_test)

# Calculate and store the accuracy of Weighted KNN
accuracy_weighted = accuracy_score(y_test, y_pred_weighted)

# Step 8: Create a Naive Bayes classifier
clf = GaussianNB()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Step 9: Make Predictions with Naive Bayes
y_pred_nb = clf.predict(X_test)

# Calculate and store the accuracy of Naive Bayes
accuracy_nb = accuracy_score(y_test, y_pred_nb)

# Step 10: Print accuracies
print(f"KNN Accuracy: {accuracy_knn * 100:.2f}%")
print(f"Weighted KNN Accuracy: {accuracy_weighted * 100:.2f}%")
print(f"Naive Bayes Accuracy: {accuracy_nb * 100:.2f}%")

# Step 11: Visualize accuracies
# Create a bar plot using Seaborn
models = ['KNN', 'Weighted KNN', 'Naive Bayes']
sns.barplot(x=models, y=[accuracy_knn, accuracy_weighted, accuracy_nb])
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.show()

# Create a pie chart using Matplotlib
labels = ['KNN', 'Weighted KNN', 'Naive Bayes']
plt.pie([accuracy_knn, accuracy_weighted, accuracy_nb], labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Accuracy Distribution Among Models')
plt.show()
