from flask import Flask, request, jsonify
import pickle as pkl
import numpy as np 
import pandas as pd


app = Flask(__name__)


@app.route('/submit', methods=['POST'])
def submit_form():
    form_data = request.get_json()

    # Extract form data
    gender = form_data.get('gender')
    SeniorCitizen = form_data.get('SeniorCitizen')
    Partner = form_data.get('Partner')
    Dependents = form_data.get('Dependents')
    PhoneService = form_data.get('PhoneService')
    MultipleLines = form_data.get('MultipleLines')
    InternetService = form_data.get('InternetService')
    OnlineSecurity = form_data.get('OnlineSecurity')
    OnlineBackup = form_data.get('OnlineBackup')
    DeviceProtection = form_data.get('DeviceProtection')
    TechSupport = form_data.get('TechSupport')
    StreamingTV = form_data.get('StreamingTV')
    StreamingMovies = form_data.get('StreamingMovies')
    Contract = form_data.get('Contract')
    PaperlessBilling = form_data.get('PaperlessBilling')
    PaymentMethod = form_data.get('PaymentMethod')
    MonthlyCharges = form_data.get('MonthlyCharges')
    TotalCharges = form_data.get('TotalCharges')
    tenure_group = form_data.get('tenure_group')

    # Create a new FormData object
    new_form_data = [
        gender,
        SeniorCitizen,
        Partner,
        Dependents,
        PhoneService,
        MultipleLines,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies,
        Contract,
        PaperlessBilling,
        PaymentMethod,
        MonthlyCharges,
        TotalCharges,
        tenure_group,
    ]
    # final_features=[np.array(new_form_data)]
    columns_names=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',
       'TotalCharges', 'tenure_group']
    df1=pd.read_csv(r'C:\Users\user\Desktop\MP2_M\new_telco.csv')
    df1=df1.drop('Unnamed: 0',axis=1)
    input_df = pd.DataFrame([new_form_data])
    input_df.columns=columns_names
    df_2 = pd.concat([df1, input_df], ignore_index = True) 
    df_combined = pd.concat([df1, input_df], ignore_index=True)
    # Apply one-hot encoding to categorical variables
    df_encoded = pd.get_dummies(df_combined, columns=[ 'gender', 'Partner', 'Dependents',
                                                      'PhoneService', 'MultipleLines', 'InternetService',
                                                      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                                      'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                                                      'PaperlessBilling', 'PaymentMethod','tenure_group'])
    new_df_encoded = df_encoded.tail(1)
    new_df_encoded=new_df_encoded.drop('Churn',axis=1)
    

# Predict using the model
    model=pkl.load(open(r'C:\Users\user\Desktop\MP2_M\hope_final.sav','rb'))
    output = model.predict(new_df_encoded)
    confidence=model.predict_proba(new_df_encoded)[:,1][0]

    if output==1:
        o1="The Customer is likely to be churned "
        o2="{}".format(round(confidence*100,2))
    else:
         o1="The Customer is likely to CONTINUE "
         o2="{}".format(round(confidence*100,2))        
# Convert the prediction to a JSON response
    return jsonify({
        'o1':o1,
        'o2':o2})
   
if __name__ == '_main_':
    app.run()