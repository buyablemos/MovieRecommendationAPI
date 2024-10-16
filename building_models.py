from sklearn.neighbors import NearestNeighbors
import joblib
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
import db
from last_user_info import LastUserInfo


def build_kNN_CF(database):

    rating_pivot = database.get_rating_pivot()
    print('Shape of this pivot table :',rating_pivot.shape)
    print(rating_pivot.head())

    nn_algo = NearestNeighbors(metric='cosine')
    nn_algo.fit(rating_pivot)

    joblib.dump(nn_algo, 'knn_model_CF.pkl')

def build_kNN_CBF(database):

    contents = database.get_movies_contents()
    print('Shape of the content table :',contents.shape)
    print(contents.head())

    nn_algo = NearestNeighbors(metric='cosine')
    nn_algo.fit(contents)

    joblib.dump(nn_algo, 'knn_model_CBF.pkl')


def build_SVD(database):
    ratings_df = database.get_ratings()

    reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

    trainset, testset = train_test_split(data, test_size=0.2)

    model = SVD()
    model.fit(trainset)

    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions)
    print(f"RMSE: {rmse}")

    joblib.dump(model, 'svd_model.pkl')


def train_models():
    database = db.Database()
    build_kNN_CF(database)
    LastUserInfo.save_last_trained_user("knn_cf")
    build_kNN_CBF(database)
    LastUserInfo.save_last_trained_user("knn_cbf")
    build_SVD(database)
    LastUserInfo.save_last_trained_user("svd")


