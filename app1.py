import flask
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# In[1]
with open('models/model.pkl','rb') as f:
    model = pickle.load(f)
    
# In[2]
app = Flask(__name__, template_folder="templates")

# In[3]
@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('index.html')
    
    if request.method == 'POST':
        Age = request.form['age']
        Ease_of_Online_Booking = request.form['ease_of_online_booking']
        Gate_location = request.form['gate_location']
        Food_and_Drink = request.form['food_and_drink']
        Online_Boarding = request.form['online_boarding']
        Seat_Comfort = request.form['seat_comfort']
        Baggage_Handling = request.form['baggage_handling']
        Inflight_Service = request.form['inflight_service']
        Cleanliness = request.form['cleanliness']
        
        input_variables = pd.DataFrame([[Age, Ease_of_Online_Booking, Gate_location, Food_and_Drink, Online_Boarding, Seat_Comfort, Baggage_Handling, Inflight_Service, Cleanliness]],
                                       columns=["age", "ease_of_online_booking", "gate_location", "food_and_drink", "online_boarding", "seat_comfort",
                                                "baggage_handling", "inflight_service", "cleanliness"], index=['Input'])
        prediction = model.predict(input_variables)[0]
        
        return render_template('index.html', original_input={'Age': Age, 'Ease of Online Booking': Ease_of_Online_Booking,
                                                             'Gate location': Gate_location, 'Food and Drink': Food_and_Drink,
                                                             'Online Boarding': Online_Boarding, 'Seat Comfort': Seat_Comfort,
                                                             'Baggage Handling': Baggage_Handling, 'Inflight Service': Inflight_Service,
                                                             'Cleanliness': Cleanliness}, result=prediction)

if __name__ == '__main__':
    app.run()

