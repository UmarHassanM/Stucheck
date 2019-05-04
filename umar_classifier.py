from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from time import time
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

def train_classify(data_dir,nrof_train_images_per_class,classifier_filename,mode='TRAIN',use_split_dataset=False,model="./model/20170511-185253.pb",image_size=160,batch_size=64,min_nrof_images_per_class=1):
    
    with tf.Graph().as_default():

        with tf.Session() as sess:

            if use_split_dataset == True:
                dataset_tmp = facenet.get_dataset(data_dir)
                train_set, test_set = split_dataset(dataset_tmp, min_nrof_images_per_class, nrof_train_images_per_class)
                if (mode=='TRAIN'):
                    dataset = train_set
                elif (mode=='CLASSIFY'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')            


            paths, labels = facenet.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            # Load the model
            print('Loading feature extraction model')
            t0=time()
            facenet.load_model(model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            t1=time()
            print('Model Loading : ',t1-t0)

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            t0=time()
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images /batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

            classifier_filename_exp = os.path.expanduser(classifier_filename)
            t1=time()
            print('Feature Extraction : ',t1-t0)

            if (mode=='TRAIN'):
                # Train classifier
                print('Training classifier')
                t0=time()
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)
                '''model = RandomForestClassifier()

                # Choose some parameter combinations to try
                parameters = {'n_estimators': [4, 6, 9], 
                              'max_features': ['log2', 'sqrt','auto'], 
                              'criterion': ['entropy', 'gini'],
                              'max_depth': [2, 3, 5, 10], 
                              'min_samples_split': [2, 3, 5],
                              'min_samples_leaf': [1,5,8]
                             }

                # Type of scoring used to compare parameter combinations
                acc_scorer = make_scorer(accuracy_score)

                # Run the grid search
                grid_obj = GridSearchCV(model, parameters, scoring=acc_scorer)
                grid_obj = grid_obj.fit(emb_array, labels)

                # Set the clf to the best combination of parameters
                model = grid_obj.best_estimator_

                # Fit the best algorithm to the data. 
                model.fit(emb_array, labels)'''
                t1=time()
                print('Training Time : ',t1-t0)
            
                # Create a list of class names
                class_names = [ cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

            elif (mode=='CLASSIFY'):
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)
                    class_names.append('Unknown')

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                #print(best_class_indices)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                #print(best_class_probabilities)

                for i in range(len(best_class_indices)):
                    if best_class_probabilities[i] > 0.300:
                        print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    else:
                        print('%4d  %s: %.3f' % (i, class_names[-1], best_class_probabilities[i]))

                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)
                
            
def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set
