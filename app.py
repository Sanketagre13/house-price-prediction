from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np

app=Flask(__name__)
data=pd.read_csv("data.csv")
data1=pd.read_csv("house_data.csv")
fdata=pd.read_csv("final_data.csv")
pipe=pickle.load(open("model1.pkl",'rb'))

@app.route("/")
def index():
    cities=sorted(data1['City'].unique())
    Chennai_location=sorted(data1[data1['City']=="Chennai"]['Location'].unique())
    Delhi_location=sorted(data1[data1['City']=="Delhi"]['Location'].unique())
    Hyderabad_location=sorted(data1[data1['City']=="Hyderabad"]['Location'].unique())
    Kolkata_location=sorted(data1[data1['City']=="Kolkata"]['Location'].unique())
    Mumbai_location=sorted(data1[data1['City']=="Mumbai"]['Location'].unique())
    Banglore_location=sorted(data1[data1['City']=="Banglore"]['Location'].unique())
    return render_template("index.html",cities=cities,Chennai_location=Chennai_location,Mumbai_location=Mumbai_location,Hyderabad_location=Hyderabad_location,Kolkata_location=Kolkata_location,Delhi_location=Delhi_location,Banglore_location=Banglore_location)

@app.route("/prediction",methods=['post'])
def prediction():
    location1=request.form.get("location")
    city=request.form.get("city")
    bhk=request.form.get("bhk")
    #print(location1,bhk,city)
    inputs=pd.DataFrame([[location1,bhk,city]],columns=['Location','No. of Bedrooms','city'])
    pred= pipe.predict(inputs)[0]
    pred=np.round(pred,2)
    print(pred)
    return str(pred)


if __name__=="__main__":
    app.run(debug=True,port=5001)
