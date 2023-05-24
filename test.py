
import pickle
import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI
import requests

app = FastAPI()

file_url = 'https://drive.google.com/uc?id=1y0ZAhithrrdK4x-KA1hHRplmGtedH3Wu'

response = requests.get(file_url)

with open('file.pkl', 'wb') as file:
    file.write(response.content)
with open('file.pkl','rb') as file:
    data= pickle.load(file)

tf_url = "https://drive.google.com/uc?id=1GS80yU1uByRfSFosloXKmJ-0NhVEqJpG"
response1 = requests.get(tf_url)

with open('tf.pkl', 'wb') as file:
    file.write(response1.content)
with open('tf.pkl','rb') as file:
    tfidf= pickle.load(file)

it_url = 'https://drive.google.com/uc?id=1Atnr2xRdL3FUxkdlQqC2tIVJVEkBOjKp'

response = requests.get(it_url)

with open('item.pkl', 'wb') as file:
    file.write(response.content)
with open(r"item.pkl",'rb') as file:
    Item_based= pickle.load(file)




@app.get("/similar_books")
def similar_books(book_name:str):
  # The book name entered is converted into lower case and removing the additional characters and additional space using regular expressions
  book_name= re.sub('[^a-zA-Z0-9]',' ',book_name.lower())
  book_name=re.sub('\s+'," ",book_name)

  # Transforming the book name into TFIDF Vectorizer
  book_vector=data.transform([book_name])

  # Computing the similarities between the book name query vector and the tfid matrix
  similarity= cosine_similarity(book_vector,tfidf).flatten()

  # After getting tje similarities, we are getting the top 10 book index similar to query using numpy arg partition
  similar_book_id = np.flip(np.argpartition(similarity,-10)[-10:])

  # With the book index of the books we are creating a dataframe
  similar_books_df = pd.DataFrame(columns=['Book-Title','Book-Author'])

  # using for loop to iterate over the similar book id list and appending the data to the similar_books_df
  for i in similar_book_id:
    similar_books_df = pd.concat([similar_books_df,pd.DataFrame(Item_based.iloc[i][['Book-Title','Book-Author']]).transpose()])
  similar_books_df.reset_index(drop=True,inplace=True)
  similar_books_df.drop_duplicates(inplace=True)
  similar_books_df['book & author'] = similar_books_df['Book-Title'] + '  -  ' + similar_books_df['Book-Author']
  # Returning the similar book Data Frame
  return similar_books_df['book & author']

