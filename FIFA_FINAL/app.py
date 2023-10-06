from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained machine learning model
model = joblib.load('model1.joblib')
le=joblib.load("label_encoder1.joblib")
# Load the label encoder for 'Winnermatch'
#label_encoder = LabelEncoder()
# label_encoder.fit(['Team_A', 'Team_B', 'Draw'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get team names from the form
    Year = (int)(request.form['Year'])
    print(Year)
    
    Stadium = (request.form['Stadium'])
    print(Stadium)
    HomeTeamName = (int)(request.form['HomeTeamName'])
    print(HomeTeamName)
    AwayTeamName = (int)(request.form['AwayTeamName'])
    print(AwayTeamName)
    Referee = (int)(request.form['Referee'])
    print(Referee)

    
    prediction = model.predict(np.array([Year,Stadium,HomeTeamName,AwayTeamName,Referee]).reshape(1,-1))
#     pred=model.predict(np.array([team1,team2]).reshape(1,-1))
#     # Decode the prediction back to 'Winnermatch' labels
    #predicted_winner = le.inverse_transform([prediction)
    predicted_winner=prediction
    print(predicted_winner)
    if(predicted_winner=='Team_A'):
        res="TeamA"
    else:
        res="TeamB"
    return render_template('result.html', predicted_winner=res)

if __name__ == '__main__':
    app.run(debug=True)

