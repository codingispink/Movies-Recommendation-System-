import numpy as np
import pandas as pd

movies=pd.read_csv('movies.csv', sep=';', encoding='latin-1').drop('Unnamed: 3', axis=1)
#print('Shape of this dataset :', movies.shape)
movies.head()

ratings = pd.read_csv('ratings.csv',sep=';')
#print('Shape of the dataset:',ratings.shape)
ratings.head()

users = pd.read_csv('users.csv', sep=';')
#print('Shape of the dataset:', users.shape)
users.head()

#Collaborative Filtering
#People who watch A movie also watch and like B movies
#Pivot Table with respect to ratings given by users to movies
rating_pivot = ratings.pivot_table(values='rating',columns='userId', index='movieId').fillna(0)
print('Shape of this pivot table :', rating_pivot.shape)
rating_pivot.head()
#RECOMMEND BASED ON COLLAB FILTERING

#ML Model training for recommending movies based on users ratings
from sklearn.neighbors import NearestNeighbors
nn_algo = NearestNeighbors(metric='cosine')
nn_algo.fit(rating_pivot)
# Developing the class of collab filtering recommendation engine
class Recommender:
    def __init__(self):
        self.hist = [] #this list will store movies that called atleast one using recommend_on_movie method
        self.ishist= False #check if history is empty
    def recommend_on_movie(self, movie, n_recommend=5):
        self.ishist = True
        movieid = int(movies[movies['title']==movie]['movieId'])
        self.hist.append(movieid)
        distance,neighbors=nn_algo.kneighbors([rating_pivot.loc[movieid]],n_neighbors=n_recommend+1)
        movieids=[rating_pivot.iloc[i].name for i in neighbors[0]]
        rcm = [str(movies[movies['movieId']==mid]['title']).split('\n')[0].split('  ')[-1] for mid in movieids if mid not in[movieid]]
        return rcm[:n_recommend]
    #this method will recommend movies based on history stored in self.hist list
    def recommend_on_history(self, n_recommend =5):
        if self.ishist == False:
            return print('No history found')
        history = np.array([list(rating_pivot.loc[mid]) for mid in self.hist])
        distance,neighbors =nn_algo.kneighbors([np.average(history, axis=0)],n_neighbors=n_recommend + len(self.hist))
        movieids = [rating_pivot.iloc[i].name for i in neighbors[0]]
        rcm =[str(movies[movies['movieId']==mid]['title']).split('\n')[0].split('  ')[-1] for mid in movieids if mid not in self.hist]
        return rcm[:n_recommend]


    
recommender = Recommender()
recommender.recommend_on_history()
print(recommender.recommend_on_movie('Father of the Bride Part II (1995)'))
print(recommender.recommend_on_history())
print(recommender.recommend_on_movie('Tigerland (2000)'))

#RECOMMEND BASED ON CONTENT BASED FILTERING
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
genres = vectorizer.fit_transform(movies.genres).toarray()
contents = pd.DataFrame(genres,columns=vectorizer.get_feature_names_out())
print('Shape of the content table:',contents.shape)
contents.head()

from sklearn.neighbors import NearestNeighbors
nn_algo = NearestNeighbors(metric='cosine')
nn_algo.fit(contents)


class Recommender: 
    def __init__(self):
        self.hist = [] #this list will store movies that called atleast one using recommend_on_movie method
        self.ishist= False #check if history is empty
    def recommend_on_movie(self, movie, n_recommend=5):
        self.ishist = True
        iloc = movies[movies['title']==movie].index[0]
        self.hist.append(iloc)
        distance,neighbors=nn_algo.kneighbors([contents.iloc[iloc]],n_neighbors=n_recommend+1)
        rcm=[movies.iloc[i]['title'] for i in neighbors[0] if i not in [iloc]]
        return rcm[:n_recommend]
    #this method will recommend movies based on history stored in self.hist list
    def recommend_on_history(self, n_recommend =5):
        if self.ishist == False:
            return print('No history found')
        history = np.array([list(contents.iloc[iloc]) for iloc in self.hist])
        distance,neighbors =nn_algo.kneighbors([np.average(history, axis=0)],n_neighbors=n_recommend + len(self.hist))
        rcm = [movies.iloc[i]['title'] for i in neighbors[0] if i not in self.hist]
        return rcm[:n_recommend]
recommender = Recommender()
#print(recommender.recommend_on_history())
histt = recommender.recommend_on_history()
print('Recommended movies based on your history are: '+str(histt))
rcmdt = recommender.recommend_on_movie('Steal This Movie! (2000)')
print('Recommended movies based on this movie are: ' +str(rcmdt))
rcmdt = recommender.recommend_on_movie('Tigerland (2000)')
print('Recommended movies based on this movie are: ' +str(rcmdt))
histt = recommender.recommend_on_history()
print('Recommended movies based on your history are: '+str(histt))
