from flask import Flask, render_template,request
import nltk
from nltk.corpus import stopwords   
import pickle 
import re
import os
from healthcheck import HealthCheck
# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'svmtfidf1.pkl'
model = pickle.load(open(filename, 'rb'))
vect= pickle.load(open('vectorizer.pkl','rb'))
# # app = Flask(__name__,template_folder='templsates')

app=Flask(__name__)

#decontract
def decontracted(phrase):
	# specific
	phrase = re.sub(r"won't", "will not", phrase)
	phrase = re.sub(r"can\'t", "can not", phrase)

	# general
	phrase = re.sub(r"n\'t", " not", phrase)
	phrase = re.sub(r"\'re", " are", phrase)
	phrase = re.sub(r"\'s", " is", phrase)
	phrase = re.sub(r"\'d", " would", phrase)
	phrase = re.sub(r"\'ll", " will", phrase)
	phrase = re.sub(r"\'t", " not", phrase)
	phrase = re.sub(r"\'ve", " have", phrase)
	phrase = re.sub(r"\'m", " am", phrase)
	return phrase

stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
			"you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
			'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
			'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 
			'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
			'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
			'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
			'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
			'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
			'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
			's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 
			've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
			"hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
			"mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 
			'won', "won't", 'wouldn', "wouldn't"])


#def clean_text(sentance):
def clean_text(sentance):
	sentance = decontracted(sentance)
	sentance = re.sub(r"\S*\d\S*", "", sentance).strip()
	sentance = re.sub('[^A-Za-z]+', ' ', sentance)
	# https://gist.github.com/sebleier/554280
	sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
	return sentance.strip()
    
print("before home")

@app.route("/")
def hello():
    print("home")
    if request.method=='GET':
        return render_template("login.html")
        #return "hello youtube"

health = HealthCheck(app, "/hcheck")
#,method=['POST']
@app.route("/predict",methods=["GET", "POST"])
def predict():
    # if request.method=='GET':
    #     return render_template("login.html")
    if request.method=='POST':
        input_text= request.form['comment']
        #input_text=input_text.reshape(1,-1)
        input_text = decontracted(input_text)
        input_text= clean_text(input_text)
        test_vect  = vect.transform([input_text])
        print("vector",test_vect)
        #print("model",model.shape)
        #print("vect",vect.shape)
        pred = model.predict(test_vect)
        print("pred",pred)
        #processed_doc1 = ' '.join([word for word in input_text.split() if word not in stop_words])

        #pred=model.predict()

        return render_template("result.html",pred=pred)
    else:
        return render_template("index.html")
   


# # if __name__ == "__main__":
# #     #app.run(host='0.0.0.0',port=8080,debug=True)
# #     app.run(debug=True)

if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.run(host='0.0.0.0', debug=True,port=int(os.environ.get('PORT', 5000)))