import nltk 
#Natural Language tool kit

# nltk.download("stopwords")
# nltk.download("wordnet")

import json
import pandas
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

review_stars=[]
review_text=[]
with open("yelp_review_part.json", encoding="utf-8") as f:
	for line in f:
		json_line = json.loads(line)
		review_stars.append(json_line["stars"])
		review_text.append(json_line["text"])

# print(review_stars)
# print(review_text)

dataset=pandas.DataFrame(data={"text": review_text, "stars": review_stars})

# print(dataset)

## We're going to subset the dataset

dataset = dataset[0:3000]
dataset= dataset[(dataset['stars'] == 1) | (dataset['stars'] == 3) | (dataset['stars'] == 5)]
#We're predicting whether a review is really bad, really good, or in the middle. 
# print(dataset)

data = dataset["text"]
target = dataset["stars"]

lemmatizer = WordNetLemmatizer()

#Preprocessing, like removing puncutation, stopwords
def pre_processing(text):
	text_processed = text.translate(str.maketrans("","", string.punctuation))
	text_processed = text_processed.split()
	result = []
	for word in text_processed:
		word_processed = word.lower()
		if word_processed not in stopwords.words("english"):
			word_processed = lemmatizer.lemmatize(word_processed)
			result.append(word_processed)
	return result

test_text = "Total bill for this horrible service? Over $8Gs. These crooks actually had the nerve to charge us $69 for 3 pills. I checked online the pills can be had for 19 cents EACH! Avoid Hospital ERs at all costs."
# print(test_text)
# print(pre_processing(test_text))

count_vectorize_transformer = CountVectorizer(analyzer=pre_processing).fit(data)

data =count_vectorize_transformer.transform(data)

machine = MultinomialNB()
machine.fit(data,target)

#Using a new dataset with text but no stars
new_reviews = pandas.read_csv("new_reviews.csv")
new_reviews_transformed = count_vectorize_transformer.transform(new_reviews.iloc[:,0])


prediction = machine.predict(new_reviews_transformed)
prediction_prob = machine.predict_proba(new_reviews_transformed)
# print(prediction)
# print(prediction_prob)


new_reviews['prediction'] = prediction
prediction_prob_dataframe = pandas.DataFrame(prediction_prob)


prediction_prob_dataframe = prediction_prob_dataframe.rename(columns={
	prediction_prob_dataframe.columns[0]: "prediction_prob_1",
	prediction_prob_dataframe.columns[1]: "prediction_prob_3",
	prediction_prob_dataframe.columns[2]: "prediction_prob_5"
	})

# print(prediction_prob_dataframe)

new_reviews = pandas.concat([new_reviews, prediction_prob_dataframe], axis=1)


new_reviews = new_reviews.rename(columns={
	new_reviews.columns[0]: "text"
	})

new_reviews['prediction'] = new_reviews['prediction'].astype(int)
new_reviews['prediction_prob_1'] = round(new_reviews['prediction_prob_1'],4)
new_reviews['prediction_prob_3'] = round(new_reviews['prediction_prob_3'],4)
new_reviews['prediction_prob_5'] = round(new_reviews['prediction_prob_5'],4)


print(new_reviews)
new_reviews.to_csv("new_reviews_with_prediction.csv", index=False)






