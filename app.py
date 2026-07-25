from flask import Flask,render_template,request 
import pickle
import pandas as pd 


df = pd.read_csv("Superstore Sales Dataset.csv", sep=";", encoding="latin1")

#Creating Flask application
app = Flask(__name__)

#Load Trained Model 
with open('Store_Sales_Data.pkl','rb') as file:
    model = pickle.load(file)
#Load Trained Scaler
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)
#Load Feature Names
with open('features.pkl','rb') as file:
    features = pickle.load(file)
#Load Encoder file
with open('encoder.pkl','rb') as file:
    encoder = pickle.load(file)


#Route for Home Page 
@app.route("/")
def home():

    states = sorted(df["State"].unique())
    cities = sorted(df["City"].unique())

    return render_template(
        "index.html",
        states=states,
        cities=cities
    )

#Route for Prediction
@app.route("/predict", methods=["POST"])
def predict():

    ship_mode = request.form["ship_mode"]
    ship_mode = encoder['Ship Mode'].transform([ship_mode])[0]
    segment = request.form["segment"]
    segment = encoder['Segment'].transform([segment])[0]
    region = request.form["region"]
    region = encoder['Region'].transform([region])[0]
    category = request.form["category"]
    category = encoder['Category'].transform([category])[0]
    sub_category = request.form["sub_category"]
    sub_category = encoder['Sub-Category'].transform([sub_category])[0]

    shipping_days = float(request.form["shipping_days"])

    order_year = int(request.form["order_year"])
    order_month = int(request.form["order_month"])
    order_day = int(request.form["order_day"])

    weekday = request.form["weekday"]
    weekday = encoder['Weekday'].transform([weekday])[0]

    state = request.form["state"]
    city = request.form["city"]

    # Create Empty DataFrame
    new_data = pd.DataFrame(0, index=[0], columns=features)

    # Fill User Inputs
    new_data["Ship Mode"] = ship_mode
    new_data["Segment"] = segment
    new_data["Region"] = region
    new_data["Category"] = category
    new_data["Sub-Category"] = sub_category
    new_data["Shipping Days"] = shipping_days
    new_data["Order Year"] = order_year
    new_data["Order Month"] = order_month
    new_data["Order Day"] = order_day
    new_data["Weekday"] = weekday

    # State One-Hot Encoding
    state_column = "State_" + state
    if state_column in new_data.columns:
        new_data[state_column] = 1

    # City One-Hot Encoding
    city_column = "City_" + city

    if city_column in new_data.columns:
        new_data[city_column] = 1
    numeric_cols = [
    "Shipping Days",
    "Order Year",
    "Order Month",
    "Order Day"
    ]

    new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])
    print(new_data.head())
    prediction = model.predict(new_data)

    return render_template(
        "index.html",
        prediction=round(prediction[0],2),
        states=sorted(df["State"].unique()),
        cities=sorted(df["City"].unique())
    )







#For hosting the model on the web page 
if __name__ == '__main__':
    app.run(debug=True)



 