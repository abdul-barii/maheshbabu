import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib
from sklearn.preprocessing import LabelEncoder

# Load your FIFA dataset here
# Replace 'FIFA.csv' with the actual path to your dataset
fifa_data = pd.read_csv('FIFA.CSV')

# Preprocess your data, including feature selection and encoding categorical variables
# Select relevant features and encode categorical columns
# Include 'HomeTeam' and 'AwayTeam' as features in your training data
le=LabelEncoder()

for i in ["Year","Stadium","HomeTeamName","AwayTeamName","Referee"]:
    fifa_data[i]=le.fit_transform(fifa_data[i])
print(fifa_data.dtypes)
print(fifa_data.head())
X = fifa_data[["Year","Stadium","HomeTeamName","AwayTeamName","Referee"]]
y = fifa_data['Winnermatch']
#print(fifa_data.dtypes)
# Identify categorical columns
# categorical_columns = ["Stage","Stadium","City",'HomeTeamName', 'AwayTeamName',"Referee","Assistant1","Assistant2","HomeTeamInitials","AwayTeamInitials"]

# # Apply one-hot encoding to the categorical columns
# X = pd.get_dummies(X, columns=categorical_columns)

# # Encode the target variable (Winnermatch)
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y)

# # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train a machine learning model (Random Forest in this example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# # Create and train a HistGradientBoostingClassifier
# # model = HistGradientBoostingClassifier(random_state=42)
# # model.fit(X_train, y_train)

# # Make predictions on the test set
y_pred = model.predict(X_test)

# # Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, 'model1.joblib')
# # Now you have a trained model that can make predictions based on input features
# # You can integrate this model into your web application to predict match outcomes

joblib.dump(le, 'label_encoder1.joblib')

