from flask import Flask, request, jsonify, render_template, make_response
import numpy as np
import requests
import csv
import pandas as pd
import flask_excel as excel
import re
from io import StringIO

pd.set_option("display.max_colwidth", 200)
import numpy as np
import json
import re
import gzip
import spacy
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
# from wordcloud import WordCloud
from textblob import TextBlob
import nltk
from nltk import FreqDist
import warnings

warnings.filterwarnings('ignore')
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('dashboard.html')


'''@app.route('/senti',methods =["GET", "POST"])
def senti():
    if request.method == 'POST':
        #print(request.files['file'])
        f = request.files['file']
        df = pd.read_excel(f,sheet_name='Feedback Details')
        review=request.form.get('Review')
        #print(review)
        df=df[df[review]!="--"].reset_index(drop=True)
        nltk.download('stopwords')
        stop_words=set(stopwords.words('english')+list(punctuation))       
        def remove_stopwords(remove_words):
            rev_new = " ".join([i for i in remove_words if i not in stop_words])
            return rev_new 
        df[review] =df[review].str.replace("n\'t", " not")
        df[review] =df[review].str.replace("[^a-zA-Z#]", " ")
        df[review]=df[review].apply(lambda x:' '.join([w for w in x.split() if len(w)>2]))
        feedback_review = [remove_stopwords(r.lower().split()) for r in df[review]]
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        def lemmatization(texts, tags=['NOUN', 'ADJ']):
            output = []
            for sent in texts:
                doc = nlp(" ".join(sent)) 
                output.append([token.lemma_ for token in doc if token.pos_ in tags])
            return output
        tokenized_reviews = pd.Series(feedback_review).apply(lambda x: x.split())
        reviews_2 = lemmatization(tokenized_reviews)
        reviews_3 = []
        for i in range(len(reviews_2)):
            reviews_3.append(' '.join(reviews_2[i]))
        df['reviews'] = reviews_3
        #Sentiment Analysis
        nltk.download('vader_lexicon')
        analyzer=SentimentIntensityAnalyzer()
        df['Negative_Score']=df[review].apply(lambda x:analyzer.polarity_scores(x)['neg'])
        df['Neutral_Score']=df[review].apply(lambda x:analyzer.polarity_scores(x)['neu'])
        df['Positive_Score']=df[review].apply(lambda x:analyzer.polarity_scores(x)['pos'])
        df['Compound_Score']=df[review].apply(lambda x:analyzer.polarity_scores(x)['compound'])
        df.loc[df['Compound_Score']>0.2,"Review_Cat"]="Postive"
        df.loc[(df['Compound_Score']>=-0.2)&(df['Compound_Score']<=0.2),"Review_Cat"]="Neutral"
        df.loc[df['Compound_Score']<-0.2,"Review_Cat"]="Negative"
        #Percentages of Sentiment Calculation
        pos_review=[j for i ,j in enumerate (df[review]) if df['Compound_Score'][i]>0.2]
        neu_review=[j for i ,j in enumerate (df[review]) if 0.2>=df['Compound_Score'][i]>=-0.2]
        neg_review=[j for i ,j in enumerate (df[review]) if df['Compound_Score'][i]< -0.2]

        print("Percentage of Positve Reviews: {}%".format(len(pos_review)*100/len(df[review])))
        print("Percentage of Neutral Reviews: {}%".format(len(neu_review)*100/len(df[review])))
        print("Percentage of Negative Reviews: {}%".format(len(neg_review)*100/len(df[review])))
        plt.figure(figsize=(18,12))
        plt.title("Review Distributions",fontsize=30,color="green",loc="center",rotation=0)
        g = plt.pie(round(df['Review_Cat'].value_counts(normalize=True)*100,2),explode=(0.055,0.055,0.055),labels=round(df['Review_Cat'].value_counts(normalize=True)*100,2).index,colors=['purple', 'blue', 'orange'],textprops={'fontsize': 20},autopct="%1.2f%%", startangle=180)
        #plt.show()
        #df2=pd.DataFrame(df)
        #selected_columns = df[["City","State",'Star Rating','Review_Cat']]
        #df2=selected_columns.copy()
        #print(df2)
        Positive_city = df.loc[df['Review_Cat']=='Postive']
        Negative_city = df.loc[df['Review_Cat']=='Negative']
        Neutral_City = df.loc[df['Review_Cat']=="Neutral"]
        city_pos = Positive_city.loc[Positive_city["Review_Cat"] == "Postive"]
        city_neu = Neutral_City.loc[Neutral_City["Review_Cat"] == "Neutral"]
        city_neg = Negative_city.loc[Negative_city["Review_Cat"] == "Negative"]
        city_list = []
        city_list.append([len(city_pos), len(city_neu),len(city_neg)])
        city_list
        converted_list = [str(element) for element in city_list]
        joined_string = ",".join(converted_list)
        negative=joined_string.split("]")[0].split(',')[2]
        positive=joined_string.split("[")[1].split(',')[0]
        neutral=joined_string.split("[")[1].split(',')[1]
        # Review ratings
        rating_1 = df.loc[df['Star Rating']==1]
        rating_2 = df.loc[df['Star Rating']==2]
        rating_3 = df.loc[df['Star Rating']==3]
        rating_4 = df.loc[df['Star Rating']==4]
        rating_5 = df.loc[df['Star Rating']==5]

        r1_pos = rating_1.loc[rating_1["Review_Cat"] == "Postive"]
        r1_neu = rating_1.loc[rating_1["Review_Cat"] == "Neutral"]
        r1_neg = rating_1.loc[rating_1["Review_Cat"] == "Negative"]
        r2_pos = rating_2.loc[rating_2["Review_Cat"] == "Postive"]
        r2_neu = rating_2.loc[rating_2["Review_Cat"] == "Neutral"]
        r2_neg = rating_2.loc[rating_2["Review_Cat"] == "Negative"]
        r3_pos = rating_3.loc[rating_3["Review_Cat"] == "Postive"]
        r3_neu = rating_3.loc[rating_3["Review_Cat"] == "Neutral"]
        r3_neg = rating_3.loc[rating_3["Review_Cat"] == "Negative"]
        r4_pos = rating_4.loc[rating_4["Review_Cat"] == "Postive"]
        r4_neu = rating_4.loc[rating_4["Review_Cat"] == "Neutral"]
        r4_neg = rating_4.loc[rating_4["Review_Cat"] == "Negative"]
        r5_pos = rating_5.loc[rating_5["Review_Cat"] == "Postive"]
        r5_neu = rating_5.loc[rating_5["Review_Cat"] == "Neutral"]
        r5_neg = rating_5.loc[rating_5["Review_Cat"] == "Negative"]

        Rating_list = []
        Rating_list.append([len(r1_pos),len(r1_neu),len(r1_neg)])
        Rating_list.append([len(r2_pos),len(r2_neu),len(r2_neg)])
        Rating_list.append([len(r3_pos),len(r3_neu),len(r3_neg)])
        Rating_list.append([len(r4_pos),len(r4_neu),len(r4_neg)])
        Rating_list.append([len(r5_pos),len(r5_neu),len(r5_neg)])
        Rating_col = pd.DataFrame(Rating_list,columns =['Positive','Neutral','Negative'])


        #return render_template('senti_analyse.html',graph=plt.show(),column_names=df2.columns.values, row_data=list(df2.values.tolist()), zip=zip)

        fig = plt.figure(figsize=(9, 4))
        subset = df[df['Review_Cat'] == "Postive"]
        sns.distplot(subset["Compound_Score"], hist=False, label="Good reviews")
        subset = df[df['Review_Cat'] == "Neutral"]
        sns.distplot(subset["Compound_Score"], hist=False, label="Good reviews")
        subset = df[df['Review_Cat'] == "Negative"]
        sns.distplot(subset["Compound_Score"], hist=False, label="Bad reviews")
        imgdata = StringIO()
        fig.savefig(imgdata, format='svg')
        imgdata.seek(0)
        data = imgdata.getvalue()

        fig1 = plt.figure(figsize=(5, 4))
        plt.pie(round(df['Review_Cat'].value_counts(normalize=True) * 100, 2), explode=(0.055, 0.055, 0.055),
        labels=round(df['Review_Cat'].value_counts(normalize=True) * 100, 2).index,
        colors=['purple', 'blue', 'orange'], textprops={'fontsize': 10}, autopct="%1.2f%%", startangle=180)

        circle = plt.Circle((0, 0), 0.7, color='white')
        p = plt.gcf()
        p.gca().add_artist(circle)
        imgdata1 = StringIO()
        fig1.savefig(imgdata1, format='svg')
        imgdata1.seek(0)
        data1 = imgdata1.getvalue()
        return render_template('feedback.html',image1=data1, image=data, positivity='The positive number of cities : '+positive,negativity='The negative number of cities : '+negative,neutralize='The neutral number of cities : '+neutral
        ,column_names=Rating_col.columns.values, row_data=list(Rating_col.values.tolist()), zip=zip)
        #df_Postive=df[df['Compound_Score']>0.2]
        #df_Negative=df[df['Compound_Score']<-0.2]'''

if __name__ == "__main__":
    app.run(debug=True)
