import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Step 2: Read the CSV file
start_time = time.time()
print("Step 2: Reading the CSV file")
data = pd.read_csv('adult.csv')  # Replace 'your_dataset.csv' with the actual file path
print("--- %s seconds ---" % (time.time() - start_time))

# Step 3: Data Preprocessing
start_time = time.time()
print("Step 3: Data Preprocessing")
selected_features = ['native-country', 'marital-status', 'gender', 'income']
data_selected = data[selected_features]

# Handle missing data and apply One-Hot Encoding to categorical variables
data_selected = pd.get_dummies(data_selected, columns=['marital-status', 'gender', 'native-country'])
print("--- %s seconds ---" % (time.time() - start_time))

# Define 'X' and 'y'
X = data_selected.drop(columns=['income'])  
y = data_selected['income']

# Step 4: Split the data into a training set and a test set
start_time = time.time()
print("Step 4: Splitting the data into a training set and a test set")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("--- %s seconds ---" % (time.time() - start_time))

# Step 5: Train the models
start_time = time.time()
print("Step 5: Training the models")
k = 5  # Choose the value of K
knn = KNeighborsClassifier(n_neighbors=k)
knn_weighted = KNeighborsClassifier(n_neighbors=k, weights='distance')  
clf = GaussianNB()

knn.fit(X_train, y_train)
knn_weighted.fit(X_train, y_train)
clf.fit(X_train, y_train)
print("--- %s seconds ---" % (time.time() - start_time))

# User input for filtering criteria
start_time = time.time()
print("User input for filtering criteria")
desired_min_age = int(input("Enter the minimum age: "))
desired_max_age = int(input("Enter the maximum age: "))
desired_education = input("Enter the desired education level: ")
desired_marital_status = input("Enter the desired marital status: ")
desired_relationship = input("Enter the desired relationship status: ")
desired_race = input("Enter the desired race: ")
desired_gender = input("Enter the desired gender: ")
desired_native_country = input("Enter the desired native country: ")
print("--- %s seconds ---" % (time.time() - start_time))

# Filter the dataset based on user input
start_time = time.time()
print("Filtering the dataset based on user input")
filtered_data = data[
    (data['age'] >= desired_min_age) & (data['age'] <= desired_max_age) &
    (data['education'] == desired_education) &
    (data['marital-status'] == desired_marital_status) &
    (data['relationship'] == desired_relationship) &
    (data['race'] == desired_race) &
    (data['gender'] == desired_gender) &
    (data['native-country'] == desired_native_country)
]
print("--- %s seconds ---" % (time.time() - start_time))

# Prompt the user to choose between using the entire dataset or the filtered dataset
start_time = time.time()
print("Prompting the user to choose between using the entire dataset or the filtered dataset")
choice = input("Do you want to use the entire dataset or the filtered dataset? (Type 'entire' or 'filtered'): ")
print("--- %s seconds ---" % (time.time() - start_time))

if choice.lower() == 'entire':
    selected_data = data
elif choice.lower() == 'filtered':
    selected_data = filtered_data
else:
    print("Invalid choice. Using the entire dataset by default.")
    selected_data = data

# Now 'selected_data' contains either the entire dataset or the filtered dataset based on user input
# You can use 'selected_data' for further analysis or modeling

# Step 6: User input for prediction using the selected dataset
start_time = time.time()
print("User input for prediction using the selected dataset")
user_input = {}
for feature in ['native-country', 'marital-status', 'gender']:
    user_input[feature] = input(f"Enter the value for {feature}: ")
print("--- %s seconds ---" % (time.time() - start_time))

# Convert the user input to a DataFrame for prediction
start_time = time.time()
print("Converting the user input to a DataFrame for prediction")
user_df = pd.DataFrame([user_input])

# Apply One-Hot Encoding to the user input
user_df = pd.get_dummies(user_df, columns=['marital-status', 'gender', 'native-country'])
print("--- %s seconds ---" % (time.time() - start_time))

# Step 7: Make predictions
start_time = time.time()
print("Making predictions")
knn_prediction = knn.predict(user_df)[0]
knn_weighted_prediction = knn_weighted.predict(user_df)[0]
nb_prediction = clf.predict(user_df)[0]
print("--- %s seconds ---" % (time.time() - start_time))

# Step 8: Display predictions
start_time = time.time()
print("Displaying predictions")
print(f"KNN Prediction: {knn_prediction}")
print(f"Weighted KNN Prediction: {knn_weighted_prediction}")
print(f"Naive Bayes Prediction: {nb_prediction}")
print("--- %s seconds ---" % (time.time() - start_time))

# Step 9: Display accuracy for each model on the selected dataset
start_time = time.time()
print("Displaying accuracy for each model on the selected dataset")
accuracy_knn = accuracy_score(y_test, knn.predict(X_test))
accuracy_weighted_knn = accuracy_score(y_test, knn_weighted.predict(X_test))
accuracy_nb = accuracy_score(y_test, clf.predict(X_test))
print(f"KNN Accuracy: {accuracy_knn * 100:.2f}%")
print(f"Weighted KNN Accuracy: {accuracy_weighted_knn * 100:.2f}%")
print(f"Naive Bayes Accuracy: {accuracy_nb * 100:.2f}%")
print("--- %s seconds ---" % (time.time() - start_time))
