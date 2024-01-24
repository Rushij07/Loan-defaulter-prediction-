from flask import Flask, render_template, request
import re
import pandas as pd
import copy
import pickle
import joblib

model = pickle.load(open('DT.pkl','rb'))
imp_enc_scale = joblib.load('imp_enc_scale')
winsor = joblib.load('winsor')
data = pd.read_csv(r"C:\Users\Rushikesh Jadhav\Downloads\DT_flask_final_v2\DT_flask_final_v2\credit.csv") 



def decision_tree(data_new):
    clean1 = pd.DataFrame(imp_enc_scale.transform(data_new), columns = imp_enc_scale.get_feature_names_out())
    clean1[['num__months_loan_duration', 'num__amount', 'num__percent_of_income','num__years_at_residence', 'num__age', 'num__existing_loans_count']] = winsor.transform(clean1[['num__months_loan_duration', 'num__amount', 'num__percent_of_income','num__years_at_residence', 'num__age', 'num__existing_loans_count']])
   
    prediction = pd.DataFrame(model.predict(clean1), columns = ['default'])
    final_data = pd.concat([prediction, data_new], axis = 1)
    return(final_data)
    
            
#define flask
app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        uploadedFile = request.files['file']
        if uploadedFile is not None :
            try:

                data = pd.read_csv(uploadedFile)
            except:
                    try:
                        data = pd.read_excel(uploadedFile)
                    except:      
                        data = pd.DataFrame()
       
        final_data = decision_tree(data)
        
        html_table = final_data.to_html(classes = 'table table-striped')
        
       
        
        return render_template("new.html", Y = f"<style>\
                    .table {{\
                        width: 50%;\
                        margin: 0 auto;\
                        border-collapse: collapse;\
                    }}\
                    .table thead {{\
                        background-color: #8f6b39;\
                    }}\
                    .table th, .table td {{\
                        border: 1px solid #ddd;\
                        padding: 8px;\
                        text-align: center;\
                    }}\
                        .table td {{\
                        background-color: #32b8b8;\
                    }}\
                            .table tbody th {{\
                            background-color: #3f398f;\
                        }}\
                </style>\
                {html_table}") 

if __name__=='__main__':
    app.run(debug = True)
