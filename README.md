#NLP -  Sentiment Analysis
#Sentiment Analysis
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.tokenize import RegexpTokenizer  #for Extracting the terms with Particular Pattern
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer

amz_rev=pd.read_csv('C:/Users/91868/Desktop/Amazon_Reviews.csv')

amz_rev["Label"]=amz_rev["Label"].map({'__label__1 ':0,'__label__2 ':1})

from sklearn.model_selection import train_test_split

y=amz_rev['Label']
amz_rev.drop(columns='Label',inplace=True)
X_train,X_test,y_train,y_test=train_test_split(amz_rev,y,test_size=0.2,random_state=42)

#next we need to tokenize,stopwords removal, stemming/Lemmatization
from nltk.tokenize import RegexpTokenizer

tokenizer=RegexpTokenizer(r'\w+')
lemmatizer=WordNetLemmatizer()
stemmer=PorterStemmer()

def preprocessing(review):
    final_tokens=' '
    tokens=tokenizer.tokenize(review)
    pure_tokens=[token.lower() for token in tokens if token.lower() not in stopwords.words('english')]
    lemmas_tokens=[lemmatizer.lemmatize(pure_token) for pure_token in pure_tokens]
    final_tokens=final_tokens.join(lemmas_tokens)
    return final_tokens
X_train['Cleaned_text']=X_train['Review'].apply(preprocessing)

X_test['Cleaned_text']=X_test['Review'].apply(preprocessing)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()
vectorizer.fit(X_train['Cleaned_text'])

X_train_Tfidf=vectorizer.transform(X_train['Cleaned_text'])

X_test_Tfidf=vectorizer.transform(X_test['Cleaned_text'])

from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score
import matplotlib.pyplot as plt

#clf=MultinomialNB().fit(X_train_Tfidf,y_train)

clf=MultinomialNB()
clf.fit(X_train_Tfidf,y_train)

y_pred=clf.predict(X_test_Tfidf)
confusion_matrix(y_test,y_pred)

y_proba_pred=clf.predict_proba(X_test_Tfidf)[::,1]
fpr,tpr,thresholds=roc_curve(y_test,y_proba_pred)
plt.plot(fpr,tpr)

auc=roc_auc_score(y_test,y_proba_pred)
auc


#Logistic Regression for the above

from sklearn.linear_model import LogisticRegression
rg=LogisticRegression()
rg.fit(X_train_Tfidf,y_train)

y_pred=rg.predict(X_test_Tfidf)
confusion_matrix(y_test,y_pred)

y_proba_pred=rg.predict_proba(X_test_Tfidf)[::,1]
fpr,tpr,thresholds=roc_curve(y_test,y_proba_pred)
plt.plot(fpr,tpr)

auc=roc_auc_score(y_test,y_proba_pred)
auc
