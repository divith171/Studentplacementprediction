import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle
import psycopg2
from flask_cors import CORS, cross_origin

app = Flask(__name__)
model = pickle.load(open('./ModelSaving/model.pkl', 'rb'))
print("Inside model")
scalar = pickle.load(open('./Scaler/Scalar.pkl', 'rb'))
print("inside scalar")

db = psycopg2.connect(host="ec2-34-232-144-162.compute-1.amazonaws.com",user="ncdbsatlrmsvev",password="020f19d0f21730fb2e82a63f41e90243705226436cfd5f6fb8c0b768365025dc",database="d7ct6d0r3lqp46")
cur = db.cursor()
cur.execute("create table if not exists placements(Gender VARCHAR(10), SSCBoard VARCHAR(10),SSCPercentage NUMERIC(5,2),"
            "HSCBoard VARCHAR(10), HSCStream VARCHAR(30), HSCPercentage NUMERIC(5,2), UGDegree VARCHAR(30), "
            "UGPercentage NUMERIC(5,2), PGSpl VARCHAR(20), PGPercentage NUMERIC(5,2), WorkExperience VARCHAR(10), ETest NUMERIC(5,2))")
db.commit()

@cross_origin()
@app.route('/', methods=['GET'])
def home():
    print("Inside home page")
    return render_template('./home.html')

@cross_origin()
@app.route('/info', methods=['GET'])
def info():
    print("Inside info page")
    return render_template('./info.html')

@cross_origin()
@app.route('/developer', methods=['GET'])
def developer():
    print("Inside home page")
    return render_template('./developer.html')

@cross_origin()
@app.route('/contact', methods=['GET'])
def contact():
    print("Inside contact page")
    return render_template('./contact.html')

@cross_origin()
@app.route('/app', methods=['GET'])
def index_page():
    print("Inside app")
    return render_template('./index.html')

@cross_origin()
@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        gen = str(request.form['gen'])
        if gen == 'Male':
            gen = 1
        else:
            gen = 0

        sscp = float(request.form['sscp'])

        sscb = str(request.form['sscb'])
        if sscb == 'Central':
            sscb = 1
        else:
            sscb = 0

        hscp = float(request.form['hscp'])

        hscb = str(request.form['hscb'])
        if hscb == 'Central':
            hscb = 1
        else:
            hscb = 0

        ugdp = float(request.form['ugdp'])

        wrkx = str(request.form['wrkx'])
        if wrkx == 'Yes':
            wrkx = 1
        else:
            wrkx = 0

        etest = float(request.form['etest'])

        pgd = str(request.form['pgd'])
        if pgd == 'Mkt&HR':
            pgd = 0
        else :
            pgd = 1

        pgdp = float(request.form['pgdp'])

        hscs = str(request.form['hscs'])
        if hscs == 'Commerce':
            hscs = 1, 0
        elif hscs == 'Science':
            hscs = 0,1
        else:
            hscs = 0, 0

        ugd = str(request.form['ugd'])
        if ugd == 'Others':
            ugd = 1, 0
        elif ugd == 'ScienceTech':
            ugd = 0, 1
        else:
            ugd = 0, 0

        cols = ([[gen, sscp, sscb, hscp, hscb, ugdp, wrkx, etest, pgd, pgdp, *hscs, *ugd]])
        print(cols)
        scl = scalar.transform(cols)
        print(scl)
        pred = model.predict(scl)
        print(pred)

        col1 = str(request.form['gen'])
        col2 = str(request.form['sscb'])
        col3 = float(request.form['sscp'])
        col4 = str(request.form['hscb'])
        col5 = str(request.form['hscs'])
        col6 = float(request.form['hscp'])
        col7 = str(request.form['ugd'])
        col8 = float(request.form['ugdp'])
        col9 = str(request.form['pgd'])
        col10 = float(request.form['pgdp'])
        col11 = str(request.form['wrkx'])
        col12 = float(request.form['etest'])
        cur.execute(f"insert into placements values{(col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12)}")
        db.commit()

        if pred == np.array(1):
            return render_template('./result.html', Prediction_text="The Candidate is Placed in a Company.")
        else:
            return render_template('./result.html', Prediction_text="The Candidate is not Placed in the Company yet.")
    else:

        return render_template('./home.html')


if __name__ == "__main__":
    app.run(debug=True)