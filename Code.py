import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from sklearn.metrics import pairwise_distances
from keras.applications import DenseNet121
from sklearn.metrics.pairwise import cosine_similarity
img_width, img_height = 224, 224

# Feature extraction using DenseNet121

def extract_features(Datapath, nb_train_samples, batch_size, filename):
    Itemcodes = [] 
    datagen = ImageDataGenerator(rescale=1. / 255) 
    # DenseNet121 - Actual model used to extract features
    # weights = imagenet means load the pretrained model on imagenet dataset
    model = applications.DenseNet121(include_top=False, weights='imagenet')
    generator = datagen.flow_from_directory(
        Datapath,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    for i in generator.filenames:
        Itemcodes.append(i[(i.find("/")+1):i.find(".")])
    # Generator feeds the images sequentially
    extracted_features = model.predict_generator(generator, nb_train_samples // batch_size)
    # Reshape the extracted_features into size(training samples received, features extracted for a single sample)
    extracted_features = extracted_features.reshape((nb_train_samples, -1))
    path_features = "./" + str(filename) + "_features.npy" 
    path_ids = "./" + str(filename) + "_ids.npy"
    # Save the extracted features and product ids to .npy files 
    np.save(open(path_features, 'wb'), extracted_features)
    np.save(open(path_ids, 'wb'), np.array(Itemcodes))

# Item - Item collaborative filtering

def get_list(product_id, num_results, dataframe ,extracted_features_path, productId_path):
    # Load the product ids and feature vectors from the .npy files
    extracted_features = np.load(extracted_features_path)
    Productids = list(np.load(productId_path))
    # Find the product id of the product we want to provide recommendations for
    doc_id = Productids.index(product_id)
    # Calculate the distance between the product to recommend and other products
    pairwise_dist = pairwise_distances(extracted_features, extracted_features[doc_id].reshape(1,-1))
    # Sort and get the indices of the products in the .npy file that are closest to the product
    indices = np.argsort(pairwise_dist.flatten())[0:num_results + 1]
    # pdists = Euclidean Distance of the recommended products and the given product
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results + 1]
    id_list = []
    # Find the product id which is present in the main product_id list and add it to a list
    for i in range(1,len(indices)):
        rows = dataframe.loc[dataframe['ProductId']==Productids[indices[i]]]
        for indx, row in rows.iterrows():
            id_list.append([row['ProductId']])
    # Return the list of recommended products
    return id_list

# User - User collaborative filtering

def user_user_collaborative_filtering(user_features, user_ids, num_recommendations, target_user_id):
    target_user_index = user_ids.index(target_user_id)
    # Uses cosine similarity on user features to determine the similarity between the given user's features and other users
    similarities = cosine_similarity(user_features, user_features[target_user_index].reshape(1, -1))
    # Finds the top similar users 
    similar_users_indices = np.argsort(similarities.flatten())[:-1]
    top_similar_users_indices = similar_users_indices[-num_recommendations:]
    # Extract the top (num_recommendations) number of users similar to the given user
    recommended_user_ids = [user_ids[index] for index in top_similar_users_indices]
    return recommended_user_ids