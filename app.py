from flask import Flask, request, render_template, make_response,redirect, url_for
# Natural Language Tool Kit
import nltk
import spacy
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from nltk import word_tokenize
#from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
#import flask_excel as excel
from wordcloud import WordCloud
import xlsxwriter
#from textblob import TextBlob
# library to clean data
#import re
#import json
from io import StringIO
import pandas as pd
import numpy as np
from io import BytesIO
pd.set_option("display.max_colwidth", 200)
from string import punctuation

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_excel("feedback-details-report.xlsx",
                       sheet_name='Feedback Details')
#review = request.form.get('Review')
    # print(review)
df = df[df['Feedback'] != "--"].reset_index(drop=True)
add_stop = ["2", "26", "'s", ".", "i", "I", "��", "say", "me", "the", "my", "myself", "we", "theword", "our", "ours",
            "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
            "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
            "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
            "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
            "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
            "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "good", "should", "now"]
nltk.download('stopwords')
stop_words = set(stopwords.words('english') + list(punctuation) + list(add_stop))
    

def frequency_words(remove_words,n):
    all_words = ' '.join([text for text in remove_words])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

        # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n=n)
    return d


def remove_stopwords(remove_words):
    rev_new = " ".join([i for i in remove_words if i not in stop_words])
    return rev_new

df['Feedback'] = df['Feedback'].str.replace("n\'t", " not")
df['Feedback'] = df['Feedback'].str.replace("[^a-zA-Z#]", " ")
df['Feedback'] = df['Feedback'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))
feedback_review = [remove_stopwords(r.lower().split()) for r in df['Feedback']]
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def lemmatization(texts, tags=['NOUN', 'ADJ']):
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output
def lemmatization_noun(texts, tags=['NOUN']):
    output = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        output.append([token.lemma_ for token in doc if token.pos_ in tags])
    return output
tokenized_reviews = pd.Series(feedback_review).apply(lambda x: x.split())
reviews_2 = lemmatization(tokenized_reviews)


#print(len(reviews_2))
reviews_3 = []
for i in range(len(reviews_2)):
    reviews_3.append(' '.join(reviews_2[i]))
df['reviews'] = reviews_3
    # Sentiment Analysis
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()
df['Negative_Score'] = df['Feedback'].apply(lambda x: analyzer.polarity_scores(x)['neg'])
df['Neutral_Score'] = df['Feedback'].apply(lambda x: analyzer.polarity_scores(x)['neu'])
df['Positive_Score'] = df['Feedback'].apply(lambda x: analyzer.polarity_scores(x)['pos'])
df['Compound_Score'] = df['Feedback'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df.loc[df['Compound_Score'] > 0.2, "Review_Cat"] = "Positive"
df.loc[(df['Compound_Score'] >= -0.2) & (df['Compound_Score'] <= 0.2), "Review_Cat"] = "Neutral"
df.loc[df['Compound_Score'] < -0.2, "Review_Cat"] = "Negative"
pos_review = [j for i, j in enumerate(df['Feedback']) if df['Compound_Score'][i] > 0.2]
neu_review = [j for i, j in enumerate(df['Feedback']) if 0.2 >= df['Compound_Score'][i] >= -0.2]
neg_review = [j for i, j in enumerate(df['Feedback']) if df['Compound_Score'][i] < -0.2]

df_Postive = df[df['Compound_Score'] > 0.2]
df_Negative = df[df['Compound_Score'] < -0.2]

positive_words = frequency_words(df_Postive['reviews'], 20)
pos_word_list = list(positive_words['word'])

reviews_21 = lemmatization_noun(pd.Series(list(positive_words["word"])).apply(lambda x: x.split()))
reviews_freq = []
for i in range(len(reviews_21)):
    if len(reviews_21[i]) != 0:
        reviews_freq.append(reviews_21[i][0])
to_merge = pd.DataFrame(reviews_freq)
to_merge.columns = ["word"]
pos_freq = positive_words.merge(to_merge, on="word", how="inner")



negative_words = frequency_words(df_Negative['reviews'], 20)
neg_word_list = list(negative_words['word'])

reviews_31 = lemmatization_noun(pd.Series(list(negative_words["word"])).apply(lambda x: x.split()))
reviews_freq1 = []
for i in range(len(reviews_31)):
    if len(reviews_31[i]) != 0:
        reviews_freq1.append(reviews_31[i][0])
to_merge = pd.DataFrame(reviews_freq1)
to_merge.columns = ["word"]
neg_freq = negative_words.merge(to_merge, on="word", how="inner")


def abbrevation():
    us_abbrev = {'AL': 'Alabama',
                 'AK': 'Alaska',
                 'AZ': 'Arizona',
                 'AR': 'Arkansas',
                 'CA': 'California',
                 'CO': 'Colorado',
                 'CT': 'Connecticut',
                 'DE': 'Delaware',
                 'FL': 'Florida',
                 'GA': 'Georgia',
                 'HI': 'Hawaii',
                 'ID': 'Idaho',
                 'IL': 'Illinois',
                 'IN': 'Indiana',
                 'IA': 'Iowa',
                 'KS': 'Kansas',
                 'KY': 'Kentucky',
                 'LA': 'Louisiana',
                 'ME': 'Maine',
                 'MD': 'Maryland',
                 'MA': 'Massachusetts',
                 'MI': 'Michigan',
                 'MN': 'Minnesota',
                 'MS': 'Mississippi',
                 'MO': 'Missouri',
                 'MT': 'Montana',
                 'NE': 'Nebraska',
                 'NV': 'Nevada',
                 'NH': 'New Hampshire',
                 'NJ': 'New Jersey',
                 'NM': 'New Mexico',
                 'NY': 'New York',
                 'NC': 'North Carolina',
                 'ND': 'North Dakota',
                 'OH': 'Ohio',
                 'OK': 'Oklahoma',
                 'OR': 'Oregon',
                 'PA': 'Pennsylvania',
                 'RI': 'Rhode Island',
                 'SC': 'South Carolina',
                 'SD': 'South Dakota',
                 'TN': 'Tennessee',
                 'TX': 'Texas',
                 'UT': 'Utah',
                 'VT': 'Vermont',
                 'VA': 'Virginia',
                 'WA': 'Washington',
                 'WV': 'West Virginia',
                 'WI': 'Wisconsin',
                 'WY': 'Wyoming',
                 'DC': 'District of Columbia',
                 'AS': 'American Samoa',
                 'GU': 'Guam',
                 'MP': 'Northern Mariana Islands',
                 'PR': 'Puerto Rico',
                 'UM': 'United States Minor Outlying Islands',
                 'VI': 'U.S. Virgin Islands'}


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/user', methods=['GET'])
def user():
    return render_template('user.html')
@app.route('/city_user', methods=['GET'])
def city_user():
    return render_template('city_user.html')

@app.route('/home1', methods=['GET'])
def home1():

    writer = pd.ExcelWriter("dinebrand.xlsx", engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Full Data', index=False)
    workbook = writer.book
    pos_freq.to_excel(writer, sheet_name="Top positive keywords", index=False)


    pos_freq_chart = workbook.add_chart({'type': 'column'})
    pos_freq_chart.add_series({'values': '=Top positive keywords!$B$2:$B$8',
                               'categories': '=Top positive keywords!$A$2:$A$8',
                               'name': " Most Positive Impacting Keywords"})
    pos_freq_chart.set_title({'name': 'Positive Impacting Keywords'})
    pos_freq_chart.set_x_axis({'name': 'Keywords'})
    pos_freq_chart.set_y_axis({'name': 'Frequency Count'})
    pos_freq_chart.set_legend({'position': 'top'})

    worksheet = writer.sheets['Top positive keywords']
    worksheet.insert_chart('J10', pos_freq_chart, {'x_offset': 25, 'y_offset': 10})



    neg_freq.to_excel(writer, sheet_name="Top negative keywords", index=False)

    neg_freq_chart = workbook.add_chart({'type': 'column'})
    neg_freq_chart.add_series({'values': '=Top negative keywords!$B$2:$B$8',
                               'categories': '=Top negative keywords!$A$2:$A$8',
                               'name': " Most negative Impacting Keywords"})
    neg_freq_chart.set_title({'name': 'Negative Impacting Keywords'})
    neg_freq_chart.set_x_axis({'name': 'Keywords'})
    neg_freq_chart.set_y_axis({'name': 'Frequency Count'})
    neg_freq_chart.set_legend({'position': 'top'})

    worksheet = writer.sheets['Top negative keywords']
    worksheet.insert_chart('J10', neg_freq_chart, {'x_offset': 25, 'y_offset': 10})



    positive_percent = (round(len(pos_review) * 100 / len(df['Feedback'])))
    negative_percent = (round(len(neu_review) * 100 / len(df['Feedback'])))
    neutral_percent = (round(len(neg_review) * 100 / len(df['Feedback'])))

    to_pie_values = [positive_percent, neutral_percent, negative_percent]
    Category = ["Positive", "Neutral", "Negative"]
    to_pie = pd.DataFrame(Category)
    to_pie.columns = ["Category"]
    to_pie['Values'] = to_pie_values

    to_pie.to_excel(writer, sheet_name="Pie Data", index=False)

    to_pie_chart = workbook.add_chart({'type': 'pie'})
    to_pie_chart.add_series({'values': '=Pie Data!$B$2:$B$4',
                             'categories': '=Pie Data!$A$2:$A$4'})
    to_pie_chart.set_title({'name': 'Reviews Distributions'})
    to_pie_chart.set_legend({'position': 'top'})

    worksheet = writer.sheets['Pie Data']
    worksheet.insert_chart('J10', to_pie_chart, {'x_offset': 25, 'y_offset': 10})


    Positive_city = df.loc[df['Review_Cat'] == 'Positive']
    Negative_city = df.loc[df['Review_Cat'] == 'Negative']
    Neutral_City = df.loc[df['Review_Cat'] == "Neutral"]

    # Review ratings
    rat_1 = df.loc[df['Star Rating'] == 1]
    rat_2 = df.loc[df['Star Rating'] == 2]
    rat_3 = df.loc[df['Star Rating'] == 3]
    rat_4 = df.loc[df['Star Rating'] == 4]
    rat_5 = df.loc[df['Star Rating'] == 5]

    r1_pos = rat_1.loc[rat_1["Review_Cat"] == "Positive"]
    r1_neu = rat_1.loc[rat_1["Review_Cat"] == "Neutral"]
    r1_neg = rat_1.loc[rat_1["Review_Cat"] == "Negative"]
    r2_pos = rat_2.loc[rat_2["Review_Cat"] == "Positive"]
    r2_neu = rat_2.loc[rat_2["Review_Cat"] == "Neutral"]
    r2_neg = rat_2.loc[rat_2["Review_Cat"] == "Negative"]
    r3_pos = rat_3.loc[rat_3["Review_Cat"] == "Positive"]
    r3_neu = rat_3.loc[rat_3["Review_Cat"] == "Neutral"]
    r3_neg = rat_3.loc[rat_3["Review_Cat"] == "Negative"]
    r4_pos = rat_4.loc[rat_4["Review_Cat"] == "Positive"]
    r4_neu = rat_4.loc[rat_4["Review_Cat"] == "Neutral"]
    r4_neg = rat_4.loc[rat_4["Review_Cat"] == "Negative"]
    r5_pos = rat_5.loc[rat_5["Review_Cat"] == "Positive"]
    r5_neu = rat_5.loc[rat_5["Review_Cat"] == "Neutral"]
    r5_neg = rat_5.loc[rat_5["Review_Cat"] == "Negative"]

    postive_rating = [len(r1_pos), len(r2_pos), len(r3_pos), len(r4_pos), len(r5_pos)]
    negative_rating = [len(r1_neg), len(r2_neg), len(r3_neg), len(r4_neg), len(r5_neg)]
    neutral_rating = [len(r1_neu), len(r2_neu), len(r3_neu), len(r4_neu), len(r5_neu)]

    Rating_list = [[len(r1_pos), len(r1_neu), len(r1_neg)], [len(r2_pos), len(r2_neu), len(r2_neg)],
                   [len(r3_pos), len(r3_neu), len(r3_neg)], [len(r4_pos), len(r4_neu), len(r4_neg)],
                   [len(r5_pos), len(r5_neu), len(r5_neg)]]

    star_df = pd.DataFrame([[Rating_list[0][0], Rating_list[0][1], Rating_list[0][2]],
                            [Rating_list[1][0], Rating_list[1][1], Rating_list[1][2]],
                            [Rating_list[2][0], Rating_list[2][1], Rating_list[2][2]],
                            [Rating_list[3][0], Rating_list[3][1], Rating_list[3][2]],
                            [Rating_list[4][0], Rating_list[4][1], Rating_list[4][2]]],
                           columns=['Positive', 'Neutral', 'Negative'])
    star_df_csv = star_df

    star_df_csv["Category"] = ["Star 1", "Star 2", "Star 3", "Star 4", "Star 5"]
    star_df_csv = star_df_csv.reindex(columns=['Category', 'Positive', 'Neutral', 'Negative'])
    star_df_csv.to_excel(writer, sheet_name="Star Rating", index=False)


    star_chart = workbook.add_chart({'type': 'column'})
    star_chart.add_series({'values': '=Star Rating!$B$2:$B$6',
                           'categories': '=Star Rating!$A$2:$A$6',
                           'name': "Positive"
                           })
    star_chart.add_series({'values': '=Star Rating!$C$2:$C$6',
                           'name': "Neutral"})
    star_chart.add_series({'values': '=Star Rating!$D$2:$D$6',
                           'name': "Negative"})
    star_chart.set_title({'name': 'Reviews based on Star Rating'})
    star_chart.set_x_axis({'name': 'No. of Stars'})
    star_chart.set_y_axis({'name': 'Reviews Count'})
    star_chart.set_legend({'position': 'top'})
    worksheet = writer.sheets['Star Rating']
    worksheet.insert_chart('G10', star_chart, {'x_offset': 25, 'y_offset': 10})


    us_abbrev=abbrevation()

    pos_city_position = df.loc[df['Review_Cat'] == 'Positive']
    state_pos = pd.DataFrame(
        pos_city_position["State"].value_counts().rename_axis('State').reset_index(name='Positive(Counts)').head(5))
    state_pos['State'] = state_pos['State'].replace(us_abbrev)


    state_pos.to_excel(writer, sheet_name='positive_states', index=False)

    neg_city_position = df.loc[df['Review_Cat'] == 'Negative']
    state_neg = pd.DataFrame(
        neg_city_position["State"].value_counts().rename_axis('State').reset_index(name='Negative(Counts)').head(5))
    state_neg['State'] = state_neg['State'].replace(us_abbrev)

    state_neg.to_excel(writer, sheet_name='Negative States', index=False)
    neu_city_position = df.loc[df['Review_Cat'] == 'Neutral']
    state_neu = pd.DataFrame(
        neu_city_position["State"].value_counts().rename_axis('State').reset_index(name='Neutral(Counts)').head(5))
    state_neu['State'] = state_neu['State'].replace(us_abbrev)

    state_neg.to_excel(writer, sheet_name='Neutral States', index=False)

    city_pos = pd.DataFrame(
        pos_city_position["City"].value_counts().rename_axis('City').reset_index(name='Counts').head(5))
    city_pos.to_excel(writer, sheet_name='Positive city', index=False)

    city_neg = pd.DataFrame(
        neg_city_position["City"].value_counts().rename_axis('City').reset_index(name='Negative(Counts)').head(5))
    city_neg.to_excel(writer, sheet_name='Negative_city', index=False)

    city_neu = pd.DataFrame(
        neu_city_position["City"].value_counts().rename_axis('City').reset_index(name='Negative(Counts)').head(5))
    city_neu.to_excel(writer, sheet_name='Neutral_city', index=False)






    # fig = plt.figure(figsize=(3.3, 2))
    Positive_Word_Cloud_Analysis = ' '.join(df_Postive['reviews'])
    wordcloud = WordCloud(width=150, background_color='white', height=100, max_words=200,
                          max_font_size=20,
                          scale=3,
                          random_state=42).generate(Positive_Word_Cloud_Analysis)

    fig1 = plt.figure(figsize=(5.6, 3), facecolor='w')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    imgdata1 = StringIO()
    fig1.savefig(imgdata1, format='svg')
    imgdata1.seek(0)
    data1 = imgdata1.getvalue()


    # Negative_Word_Cloud_Analysis
    Negative_Word_Cloud_Analysis = ' '.join(df_Negative['reviews'])
    wordcloud = WordCloud(width=150, background_color='white', height=100, max_words=100,
                          max_font_size=20,
                          scale=3,
                          random_state=42).generate(Negative_Word_Cloud_Analysis)

    fig2 = plt.figure(figsize=(5.6, 3), facecolor='w')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    imgdata2 = StringIO()
    fig2.savefig(imgdata2, format='svg')
    imgdata2.seek(0)
    data2 = imgdata2.getvalue()





    writer.close()

    return render_template('dashboard.html', image1=data1, image2=data2,pos_freq_count=list(pos_freq['count'].values.tolist())[0:9],pos_freq_word=list(pos_freq['word'].values.tolist())[0:9],
                           column_names_neg_state=city_neg.columns.values,neg_freq_count=list(neg_freq['count'].values.tolist())[0:9],neg_freq_word=list(neg_freq['word'].values.tolist())[0:9],
                           row_data_neg_state=list(city_neg.values.tolist()),
                           zip2=zip,
                           star_pos=postive_rating,star_neu=neutral_rating,star_neg=negative_rating,
                           pos_per=positive_percent,
                           neg_per=negative_percent,
                           neu_per=neutral_percent,
                           column_names1=state_neg.columns.values,
                           row_data1=list(state_neg.values.tolist()), zip1=zip,
                           Positive_city=len(Positive_city),column_names2=state_neu.columns.values,
                           row_data2=list(state_neu.values.tolist()), zip4=zip,
                           Negative_city=len(Negative_city),
                           Neutral_City=len(Neutral_City), column_names=state_pos.columns.values,
                           row_data=list(state_pos.values.tolist()), zip=zip,column_names_city_pos1=city_neu.columns.values,
                           row_data_city_pos1=list(city_neu.values.tolist()), zip5=zip,
                           column_names_city_pos=city_pos.columns.values,
                           row_data_city_pos=list(city_pos.values.tolist()), zip3=zip)


@app.route('/dashboard', methods=['GET'])
def dashboard():
    return redirect(url_for('home1'))

# @app.route('/download', methods=["GET", "POST"])
# def download():
#     response = make_response(open('dinebrand.xlsx', 'rb').read())
#     response.headers['Content-Type'] = 'text/xlsx'
#     response.headers["Content-Disposition"] = "attachment; filename=DineBrand.xlsx"
#     return response

@app.route('/state_based', methods=["GET", "POST"])
def state_based():

    state = request.form.get('cityname').upper()
    select_state = df[(df['State'] == state)]
    select_state_neutral = len(select_state[df['Review_Cat'] == 'Neutral'])
    select_state_positive = len(select_state[df['Review_Cat'] == 'Positive'])
    select_state_negative = len(select_state[df['Review_Cat'] == 'Negative'])

    #
    pos_percent = (round((select_state_positive) * 100 / len(select_state['Feedback'])))
    neg_percent = (round((select_state_negative) * 100 / len(select_state['Feedback'])))
    neu_percent = (round((select_state_neutral) * 100 / len(select_state['Feedback'])))

    df_Postive_state = select_state[select_state['Compound_Score'] > 0.2]
    df_Negative_state = select_state[select_state['Compound_Score'] < -0.2]
    Service = " ".join(df_Postive_state.loc[df_Postive_state['State'] == state, 'reviews'])



    positive_words = frequency_words(df_Postive_state['reviews'], 20)
    pos_word_list = list(positive_words['word'])

    reviews_21 = lemmatization_noun(pd.Series(list(positive_words["word"])).apply(lambda x: x.split()))
    reviews_freq = []
    for i in range(len(reviews_21)):
        if len(reviews_21[i]) != 0:
            reviews_freq.append(reviews_21[i][0])
    to_merge1 = pd.DataFrame(reviews_freq)
    to_merge1.columns = ["word"]
    pos_freq = positive_words.merge(to_merge1, on="word", how="inner")


    negative_words = frequency_words(df_Negative['reviews'], 20)
    neg_word_list = list(negative_words['word'])

    reviews_31 = lemmatization_noun(pd.Series(list(negative_words["word"])).apply(lambda x: x.split()))
    reviews_freq1 = []
    for i in range(len(reviews_31)):
        if len(reviews_31[i]) != 0:
            reviews_freq1.append(reviews_31[i][0])
    to_merge1 = pd.DataFrame(reviews_freq1)
    to_merge1.columns = ["word"]
    neg_freq = negative_words.merge(to_merge1, on="word", how="inner")




    rat_1 = select_state.loc[select_state['Star Rating'] == 1]
    rat_2 = select_state.loc[select_state['Star Rating'] == 2]
    rat_3 = select_state.loc[select_state['Star Rating'] == 3]
    rat_4 = select_state.loc[select_state['Star Rating'] == 4]
    rat_5 = select_state.loc[select_state['Star Rating'] == 5]

    r1_pos = rat_1.loc[rat_1["Review_Cat"] == "Positive"]
    r1_neu = rat_1.loc[rat_1["Review_Cat"] == "Neutral"]
    r1_neg = rat_1.loc[rat_1["Review_Cat"] == "Negative"]
    r2_pos = rat_2.loc[rat_2["Review_Cat"] == "Positive"]
    r2_neu = rat_2.loc[rat_2["Review_Cat"] == "Neutral"]
    r2_neg = rat_2.loc[rat_2["Review_Cat"] == "Negative"]
    r3_pos = rat_3.loc[rat_3["Review_Cat"] == "Positive"]
    r3_neu = rat_3.loc[rat_3["Review_Cat"] == "Neutral"]
    r3_neg = rat_3.loc[rat_3["Review_Cat"] == "Negative"]
    r4_pos = rat_4.loc[rat_4["Review_Cat"] == "Positive"]
    r4_neu = rat_4.loc[rat_4["Review_Cat"] == "Neutral"]
    r4_neg = rat_4.loc[rat_4["Review_Cat"] == "Negative"]
    r5_pos = rat_5.loc[rat_5["Review_Cat"] == "Positive"]
    r5_neu = rat_5.loc[rat_5["Review_Cat"] == "Neutral"]
    r5_neg = rat_5.loc[rat_5["Review_Cat"] == "Negative"]

    postive_rating = [len(r1_pos), len(r2_pos), len(r3_pos), len(r4_pos), len(r5_pos)]
    negative_rating = [len(r1_neg), len(r2_neg), len(r3_neg), len(r4_neg), len(r5_neg)]
    neutral_rating = [len(r1_neu), len(r2_neu), len(r3_neu), len(r4_neu), len(r5_neu)]


    wordcloud = WordCloud(width=200, background_color='white', height=100, max_words=200,
                          max_font_size=10,
                          scale=3,
                          random_state=42).generate(Service)

    fig1 = plt.figure(figsize=(5.6, 3), facecolor='w')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    imgdata1 = StringIO()
    fig1.savefig(imgdata1, format='svg')
    imgdata1.seek(0)
    data1 = imgdata1.getvalue()

    # Negative_Word_Cloud_Analysis
    Service1 = " ".join(df_Negative_state.loc[df_Negative_state['State'] == state, 'reviews'])
    wordcloud = WordCloud(width=200, background_color='white', height=100, max_words=50,
                          max_font_size=10,
                          scale=3,
                          random_state=42).generate(Service1)

    fig2 = plt.figure(figsize=(5.6, 3), facecolor='w')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    imgdata2 = StringIO()
    fig2.savefig(imgdata2, format='svg')
    imgdata2.seek(0)
    data2 = imgdata2.getvalue()

    return render_template('state.html', Positive_city=select_state_positive, Negative_city=select_state_negative,statename=state,
                           Neutral_City=select_state_neutral, image1=data1, image2=data2,
                           star_pos=postive_rating, star_neu=neutral_rating, star_neg=negative_rating,
                           pos_per=pos_percent,pos_freq_count=list(pos_freq['count'].values.tolist())[0:9],pos_freq_word=list(pos_freq['word'].values.tolist())[0:9],
                           neg_per=neg_percent,neg_freq_count=list(neg_freq['count'].values.tolist())[0:9],neg_freq_word=list(neg_freq['count'].values.tolist())[0:9],
                           neu_per=neu_percent)
    # print(select_city)
@app.route('/table', methods=['GET'])
def table():
    data_table=df.copy()
    data_table.drop(['URL', 'Original Feedback Language (if not English)','Engagement Response','Negative_Score','Neutral_Score','Positive_Score','Compound_Score'], inplace=True, axis=1)
    return render_template('tables.html',column_names=data_table.columns.values, row_data=list(data_table.values.tolist()), zip=zip)

@app.route('/city_based', methods=["GET", "POST"])
def city_based():

    city = request.form.get('city_place')
    select_city = df[(df['City'] == city)]
    select_city_neutral = len(select_city[df['Review_Cat'] == 'Neutral'])
    select_city_positive = len(select_city[df['Review_Cat'] == 'Positive'])
    select_city_negative = len(select_city[df['Review_Cat'] == 'Negative'])

    rat_1 = select_city.loc[select_city['Star Rating'] == 1]
    rat_2 = select_city.loc[select_city['Star Rating'] == 2]
    rat_3 = select_city.loc[select_city['Star Rating'] == 3]
    rat_4 = select_city.loc[select_city['Star Rating'] == 4]
    rat_5 = select_city.loc[select_city['Star Rating'] == 5]

    r1_pos = rat_1.loc[rat_1["Review_Cat"] == "Positive"]
    r1_neu = rat_1.loc[rat_1["Review_Cat"] == "Neutral"]
    r1_neg = rat_1.loc[rat_1["Review_Cat"] == "Negative"]
    r2_pos = rat_2.loc[rat_2["Review_Cat"] == "Positive"]
    r2_neu = rat_2.loc[rat_2["Review_Cat"] == "Neutral"]
    r2_neg = rat_2.loc[rat_2["Review_Cat"] == "Negative"]
    r3_pos = rat_3.loc[rat_3["Review_Cat"] == "Positive"]
    r3_neu = rat_3.loc[rat_3["Review_Cat"] == "Neutral"]
    r3_neg = rat_3.loc[rat_3["Review_Cat"] == "Negative"]
    r4_pos = rat_4.loc[rat_4["Review_Cat"] == "Positive"]
    r4_neu = rat_4.loc[rat_4["Review_Cat"] == "Neutral"]
    r4_neg = rat_4.loc[rat_4["Review_Cat"] == "Negative"]
    r5_pos = rat_5.loc[rat_5["Review_Cat"] == "Positive"]
    r5_neu = rat_5.loc[rat_5["Review_Cat"] == "Neutral"]
    r5_neg = rat_5.loc[rat_5["Review_Cat"] == "Negative"]

    postive_rating = [len(r1_pos), len(r2_pos), len(r3_pos), len(r4_pos), len(r5_pos)]
    negative_rating = [len(r1_neg), len(r2_neg), len(r3_neg), len(r4_neg), len(r5_neg)]
    neutral_rating = [len(r1_neu), len(r2_neu), len(r3_neu), len(r4_neu), len(r5_neu)]
    try:
        plus_percent = round((select_city_positive * 100 / len(select_city)))
    except ZeroDivisionError:
        plus_percent=0
    try:
        minus_percent = round((select_city_negative *100/ len(select_city)))
    except ZeroDivisionError:
        minus_percent = 0
    try:
        middle_percent = round((select_city_neutral*100 / len(select_city)))

    except ZeroDivisionError:
        middle_percent = 0
    print(len(select_city))

    df_Postive = df[df['Compound_Score'] > 0.2]
    df_Negative = df[df['Compound_Score'] < -0.2]


    df_Postive_state = select_city[select_city['Compound_Score'] > 0.2]
    df_Negative_state = select_city[select_city['Compound_Score'] < -0.2]
    Service = " ".join(df_Postive_state.loc[df_Postive_state['City'] == city, 'reviews'])

    positive_words = frequency_words(df_Postive_state['reviews'], 20)
    pos_word_list = list(positive_words['word'])

    reviews_21 = lemmatization_noun(pd.Series(list(positive_words["word"])).apply(lambda x: x.split()))
    reviews_freq = []
    for i in range(len(reviews_21)):
        if len(reviews_21[i]) != 0:
            reviews_freq.append(reviews_21[i][0])
    to_merge2 = pd.DataFrame(reviews_freq)
    to_merge2.columns = ["word"]
    pos_freq = positive_words.merge(to_merge2, on="word", how="inner")

    negative_words = frequency_words(df_Negative_state['reviews'], 20)
    neg_word_list = list(negative_words['word'])

    reviews_31 = lemmatization_noun(pd.Series(list(negative_words["word"])).apply(lambda x: x.split()))
    reviews_freq1 = []
    for i in range(len(reviews_31)):
        if len(reviews_31[i]) != 0:
            reviews_freq1.append(reviews_31[i][0])
    to_merge2 = pd.DataFrame(reviews_freq1)
    to_merge2.columns = ["word"]
    neg_freq = negative_words.merge(to_merge2, on="word", how="inner")

    wordcloud = WordCloud(width=200, background_color='white', height=100, max_words=100,
                          max_font_size=10,
                          scale=3,
                          random_state=42).generate(Service)

    fig1 = plt.figure(figsize=(5.6, 3), facecolor='green')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    imgdata1 = StringIO()
    fig1.savefig(imgdata1, format='svg')
    imgdata1.seek(0)
    data1 = imgdata1.getvalue()

    # Negative_Word_Cloud_Analysis
    Service1 = " ".join(df_Negative_state.loc[df_Negative_state['City'] == city, 'reviews'])
    wordcloud = WordCloud(width=200, background_color='white', height=100, max_words=25,
                          max_font_size=10,
                          scale=3,
                          random_state=42).generate(Service1)

    fig2 = plt.figure(figsize=(5.6, 3), facecolor='red')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    imgdata2 = StringIO()
    fig2.savefig(imgdata2, format='svg')
    imgdata2.seek(0)
    data2 = imgdata2.getvalue()

    return render_template('city.html', Positive_city=select_city_positive, Negative_city=select_city_negative,
                           Neutral_City=select_city_neutral,pos_freq_count=list(pos_freq['count'].values.tolist())[0:9],pos_freq_word=list(pos_freq['word'].values.tolist())[0:9],
                           neg_freq_count=list(neg_freq['count'].values.tolist())[0:9],neg_freq_word=list(neg_freq['word'].values.tolist())[0:9],
                            image1=data1, image2=data2,pos_per=plus_percent,neg_per=minus_percent,neu_per=middle_percent,
                           star_pos=postive_rating, star_neu=neutral_rating, star_neg=negative_rating,
                           positive_percent=plus_percent, negative_percent=minus_percent,city=city,
                           neutral_percent=middle_percent)
    # print(select_city_negative, select_city_positive, select_city_neutral)


if __name__ == "__main__":
    app.run(debug=True)
