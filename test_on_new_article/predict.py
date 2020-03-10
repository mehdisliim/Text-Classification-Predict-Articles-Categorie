from joblib import load
from keras.models import load_model
import pandas as pd, numpy as np, os
from keras.preprocessing.text import text_to_word_sequence
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import pathlib
stopwords = stopwords.words('english')
main_path = str(pathlib.Path().absolute())
main_path = main_path[:main_path.index('test_on_new_article')]
os.chdir(main_path+'test_on_new_article/model_and_transformers/')
ohot_enc, vectorizer, model = load('OneHotEncoder.SKLEARN'), load('CountVectorizer.SKLEARN'), load_model('model.KERAS')
os.chdir(main_path+'test_on_new_article/')
def predict(article_name='insert_the_article_text_here.txt'):
    try:
        sent = open(article_name).read().replace('\n', ' ')
    except:
        sent = ' '.join([df.columns[0]]+ df[df.columns[0]].tolist())
    sent = text_to_word_sequence(sent,  filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n0123456789')
    sent = [word for word in sent if word not in stopwords]
    sent = [WordNetLemmatizer().lemmatize(word) for word in sent]
    sent = [PorterStemmer().stem(word) for word in sent]
    sent = vectorizer.transform(sent).toarray()
    vec = sent.sum(axis=0)
    vec = list(vec) + [0 for i in range(59)]
    vec = np.array(vec).reshape(1, 10, 1900)
    y_pred = model.predict(np.c_[vec])
    category = ohot_enc.inverse_transform(np.array(y_pred))
    print("The Article's Category is : "+str(category[0][0].capitalize())) 