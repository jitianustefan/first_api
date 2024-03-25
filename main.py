# -*- coding: utf-8 -*-

#import baza de date


#dataset['Translated Description'] = dataset['PrimaryColor'].apply(lambda x: translator.translate(x, dest='ro').text)



'''
dataset.describe().T

#dataset = dataset[['Title','Director','Actors','Plot','Genre']]
dataset['Description'].value_counts()[0:10].plot(kind='barh', figsize=[8,5], fontsize=15, color='navy').invert_yaxis()
print(dataset['Description'].value_counts()[0:10])
'''



#import nltk 
#elimin cuvinte pe care nu vreau sa le includ in text ul curatat care nu sunt relevante exeplu the, a of
#nltk.download('stopwords')


from fastapi import FastAPI
from pydantic import BaseModel
from typing import List



app = FastAPI()

class RecomendedItems(BaseModel):
    mesaj:str
    
    
class RecommendationResponse(BaseModel):
    recommendations: List[str]

@app.post('/')
async def response_endpoint(item: RecomendedItems):
    from rake_nltk import Rake
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity


    dataset = pd.read_csv('bd_simpla.csv')
    dataset = dataset.head(500)
    
    import re
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer #elimina conjugarile verbelor si le aduce la prezent(reducem reduntantele pentru ca inseamna acelasi lucru indiferent de conjucare in cazul reviwurilor)
    #corpus = [] # lista ce contine toate review-urile curatate
    for i in range(0,500):
        #review = re.sub('[^a-zA-Z]', ' ', dataset['Description'][i]) #reduc toate semnele de punctuatie
        if pd.notna(dataset['Description'][i]):  # Check for non-missing values
          descriere = re.sub('[^a-zA-Z]', ' ', dataset['Description'][i])
        else:
          descriere = ""
        descriere = descriere.lower()
        descriere = descriere.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        descriere = [ps.stem(word) for word in descriere if not word in set(all_stopwords)]
        descriere = ' '.join(descriere)
        dataset.loc[i, 'Description'] = descriere
        
    #CURAT TEXTUL PENTRU NUMELE PRODUSULUI de semne de punctuatie si simboluri & ...
        if pd.notna(dataset['ProductName'][i]):  # Check for non-missing values
            nume_prod = re.sub('[^a-zA-Z]', ' ', dataset['ProductName'][i])
        else:
            nume_prod = ""
        nume_prod = nume_prod.lower()
        nume_prod = nume_prod.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        nume_prod = [ps.stem(word) for word in nume_prod if not word in set(all_stopwords)]
        nume_prod = ' '.join(nume_prod)
        dataset.loc[i, 'ProductName'] = nume_prod
        #dataset['Description'][i] = review
        
    #TRADUC INPUTUL
    from deep_translator import MyMemoryTranslator
    mesaj = item.mesaj
    #input_text = input("Introduceti textul de tradus din romana in engleza: ")
    
    
    
    translated = MyMemoryTranslator(source="ro-RO", target="en-GB").translate(text=mesaj)
    #print("Textul tradus în limba engleză este:",translated)
    
    
    #!!! PRELUCREZ INPUTUL
    
    if pd.notna(translated):  # Check for non-missing values
      input_text_en = re.sub('[^a-zA-Z]', ' ', translated)
    else:
      input_text_en = ""
    input_text_en = input_text_en.lower()
    input_text_en = input_text_en.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    input_text_en = [ps.stem(word) for word in input_text_en if not word in set(all_stopwords)]
    input_text_en = ' '.join(input_text_en)
    
          
    valoarea_input_dataset = pd.DataFrame({
        'ProductName': "Text utilizator",
        'Description': [input_text_en]
        })
    
    dataset = pd.concat([valoarea_input_dataset , dataset]).reset_index(drop=True)
    
    #EXTRAG CUVINTELE CHEIE DIN DESCRIERE
    dataset['Key_words'] = ''
    r = Rake()
    
    for index, row in dataset.iterrows():
        r.extract_keywords_from_text(row['Description'])
        key_words_dict_scores = r.get_word_degrees()
        key_words_list = list(key_words_dict_scores.keys())
        dataset.at[index, 'Key_words'] = key_words_list
        
    
    #df['Genre'] = df['Genre'].apply(lambda x: [genre.lower().replace(' ','') for genre in x]) # Aceasta linie poate fi folosita pentru a modifica to lower textul din coloane
    
    
    #CREEZ BAG OF WORDS
    #dataset = dataset[['ProductName','Key_words']]
    #print(dataset['ProductName'][1], "\n", dataset['Key_words'][1])
    
    
    dataset['Bag_of_words'] = ''
    columns = ['Key_words']
    for index, row in dataset.iterrows():
        words = ''
        for col in columns:
            words += ' '.join(row[col]) + ' '
        dataset.at[index, 'Bag_of_words'] = words
        
    dataset = dataset[['ProductName','Bag_of_words']]
    
    
    from sklearn.feature_extraction.text import CountVectorizer
    count = CountVectorizer()
    count_matrix = count.fit_transform(dataset['Bag_of_words'])
    count_matrix
    
    
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    #print(cosine_sim)
    
    
    '''
    count_matrix_in = count.fit_transform([input_text_en])#aici trebuie sa adaug textul preprocesat
    
    cosine_sim_input = cosine_similarity(count_matrix_in, count_matrix_in)
    '''
    
    
    #EXECUT SI TESTEZ MODELUL DE RECOMANDARE
    # this function takes in a movie title as input and returns the top 10 recommended (similar) movies
    
    #EXECUT SI TESTEZ MODELUL DE RECOMANDARE
    # this function takes in a movie title as input and returns the top 10 recommended (similar) movies
    
    indices = pd.Series(dataset['ProductName'])
    indices[:5]
    
    
    def recommend(title, cosine_sim = cosine_sim):
        recommended_movies = []
        idx = indices[indices == title].index[0]   # to get the index of the movie title matching the input movie
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)   # similarity scores in descending order
        top_10_indices = list(score_series.iloc[1:6].index)   # to get the indices of top 10 most similar movies
        # [1:11] to exclude 0 (index 0 is the input movie itself)
        
        for i in top_10_indices:   # to append the titles of top 10 similar movies to the recommended_movies list
            recommended_movies.append(list(dataset['ProductName'])[i])
            
        return recommended_movies
    
    recommendations = recommend('Text utilizator')
    
    return RecommendationResponse(recommendations=recommendations)
