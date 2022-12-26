import base64
import os

from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import numpy as np
import pandas as pd
import pickle
import cv2 as cv

model = pickle.load(open('model/model.sav', 'rb'))


def vectorize(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    _, descriptors = sift.detectAndCompute(gray, None)
    classes = model.predict(descriptors)
    hist, _ = np.histogram(classes, np.arange(1024))
    return hist / hist.sum()


def db_create(image_dir):
    vectors, links = [], []
    for image in os.listdir(image_dir):
        if image.endswith(".jpg"):
            vectors.append(base64.b64encode((vectorize(cv.imread('images/' + image)))))
            links.append(image)
    return pd.DataFrame({"vector": vectors, "link": links})


def get_k_neighbours(vector, df, count_of_neighbours):
    neigh = NearestNeighbors(n_neighbors=count_of_neighbours, metric=lambda a, b: distance.cosine(a, b))
    neigh.fit(df['vector'].to_numpy().tolist())
    return neigh.kneighbors([vector], count_of_neighbours, return_distance=False)


def get_neighbours_links(df, neighbors):
    similar = df.iloc[neighbors[0]]
    return similar['link'].to_numpy().tolist()

db = db_create('images')
db.to_csv('out.csv')