_I would like to give credits of this project to Sachin Sarkar for inspiring the project as well as creating the dataset._
# MOVIE RECOMMENDATION SYSTEM
### INSTALLATION
Make sure you install these packages:
1. numpy
2. pandas
3. scikit-learn


### CODE ANALYSIS
**1. Collaborative Filtering**
  
  Collaborative Filtering recommends movies to users based on the interests and reactions of other users. It will first search a large group of people and identify users with similar reactions to a particular users. You will do that with the NearestNeighbors from sklearn.neighbors. Make sure you install the scikit package first.
NearestNeighbors conducts unsupervised nearest neighbor learning. The metrics to be used in this part will be 'cosine'. This metrics will measure data points with high similarity and use it to determine classification. In this project, cosine metrics will compare users that are the most similar and cluster them into a group.
```
from sklearn.neighbors import NearestNeighbors
nn_algo = NearestNeighbors(metric='cosine')
nn_algo.fit(rating_pivot)
```

**a. Recommend movies based on a specific movie**

First, we will recommend movies to users based on a specific movie. When we suggest a specific movie, the system will go through the movie list and take the movie ID of that specific movie and append to self.hist list.

```
 def recommend_on_movie(self, movie, n_recommend=5):
        self.ishist = True
        movieid = int(movies[movies['title']==movie]['movieId'])
        self.hist.append(movieid)
```
Applying KNN method, the formula is as follows: kneighbors(X, n_neighbors=None, return_distance=True) where X are points/query points, n_neighbors are numbers of neighbors required for the same, return_distance is bool. The function will take the movieid passed through and return the 5 (or a user inputed amount of) movie ids that are the nearest neighbors to this one.
```
distance,neighbors=nn_algo.kneighbors([rating_pivot.loc[movieid]],n_neighbors=n_recommend+1)
```
Afterwards, we can process and clean the data a bit so that it returns only the title of the movies. 

**b. Recommend movies based on the history**
As mentioned above, whenever we pass through  movie, the movie will be appended to the self.ishist list. Based on that specific movie, the system will recommend 5 more movies that the users are likely to be interested in. When we recommend movies based on the history, the system will look through the history of the recommended movies and suggest the movies based on this history. 

If there is no movie to be passed through, then there will be no history for the system to loop through and suggest. Therefore, if the self.ishist is empty, there is "no history found".
```
 if self.ishist == False:
return print('No history found')
```

We create a variable called history that contains only the ratings data for the movies in our self.hist list. Then, we will apply the KNN method on the average of all the data points in the history array and return 5 recommended movies based on the history. 

```
history = np.array([list(rating_pivot.loc[mid]) for mid in self.hist])
distance,neighbors =nn_algo.kneighbors([np.average(history, axis=0)],n_neighbors=n_recommend + len(self.hist))
```

**Oberseve the results**
We will now initialize the Recommender object.
```
recommender = Recommender()
```
Print the result of recommend_on_history and the first result should be "No history found"
```
print(recommender.recommend_on_history())
```
Print the result of recommend_on_movie of a specific movie (in this case: Father of the Bride Part II (1995) and it should return the 5 movies that are most likely to be watched by this type of users. Try this out multiple times with different movies.

```
print(recommender.recommend_on_movie('Father of the Bride Part II (1995)'))
```
The results returned: _['Home Alone (1990)', 'Home Alone 2', 'Mighty Ducks, The (1992)', 'Mrs. Doubtfire (1993)', 'Liar Liar (1997)']_

Print the result of recommend_on history again and it should give you a list of movies recommended to this type of users based on the history. 

**2. Recommend based on Content-based Filtering**

Content-based filtering will recommend movies to users based on the movies that the users previously watched and liked, or based on users explicit feedback.

The first example would be to recommend users movies in genres of those they like to watch before. We will create this system with the CountVectorizer from sklearn.feature_extraction.text. CountVectorizer is used to "transform a given text into a vectorizer on the basis of the frequency of each word that occurs in the entire text"[1]. When counting the frequency of each word, remember to set the stopwords so it won't count any connector word (a, an, the...) It then will transform a collection of text from the genres into a numerical matrix with fit_transform. We will also create a dataframe called Contents. In this data frame, it counts all of the feature names in the data list.
```
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
genres = vectorizer.fit_transform(movies.genres).toarray()
contents = pd.DataFrame(genres,columns=vectorizer.get_feature_names_out())
```

We will also recommend movies to users based on the movies content using the same old method KNN.

```
from sklearn.neighbors import NearestNeighbors
nn_algo = NearestNeighbors(metric='cosine')
nn_algo.fit(contents)
```
**a. Recommend based on movie** 

The rest of this method is very similar to the previous method. However, there are some minor differences. When users input a specific movie, the system will go into the content dataframe, look for keywords that are most repeated under the movie id and use it to calculate the 5 movies that are most likely liked by users watching this specific movie.

```
distance,neighbors=nn_algo.kneighbors([contents.iloc[iloc]],n_neighbors=n_recommend+1)
````

**b. Recommend based on history**
Similarly, create a variable called history containg only the genre data for the movies in our self.hist list. Then, we will apply the KNN method on the average of all the data points in the history array and return 5 recommended movies based on the history. 

```
history = np.array([list(contents.iloc[iloc]) for iloc in self.hist])
distance,neighbors =nn_algo.kneighbors([np.average(history, axis=0)],n_neighbors=n_recommend + len(self.hist))
```

**Oberseve the results**

```
print(recommender.recommend_on_movie('Father of the Bride Part II (1995)'))
```

The results returned: _['Waiting for Guffman (1996)', 'Jimmy Hollywood (1994)', 'Kolya (1996)', 'Life with Mikey (1993)', '8 1/2 Women (1999)']_


### ACKNOLWEDGEMENT
[1] https://www.geeksforgeeks.org/using-countvectorizer-to-extracting-features-from-text/

[2] https://www.kaggle.com/code/sachinsarkar/movielens-movie-recommendation-system

