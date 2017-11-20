#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import csv
import logging

import pandas
from sklearn.dummy import DummyClassifier

import lib


def extract():
    logging.info('Begin extract')
    observations = pandas.read_csv(lib.get_conf('train_path'))
    lib.archive_dataset_schemas('extract', locals(), globals())
    logging.info('End extract')
    return observations


def transform(observations):
    logging.info('Begin transform')

    observations.columns = map(lambda x: '_'.join(x.lower().split()), observations.columns)
    observations['lat'] = observations['location_1'].apply(lambda x: eval(x)[0])
    observations['long'] = observations['location_1'].apply(lambda x: eval(x)[1])

    # Dummy out response variable
    label_encoder = lib.create_label_encoder(observations['offense'])
    observations['response'] = observations['offense'].apply(lambda x: label_encoder[x])
    observations['is_grand_larceny'] = observations['offense'].apply(lambda x: x == 'GRAND LARCENY')
    logging.info('is_grand_larceny value counts: {}'.format(observations['is_grand_larceny'].value_counts()))

    lib.archive_dataset_schemas('transform', locals(), globals())
    logging.info('End transform')
    return observations


def model(observations):
    logging.info('Beginning model')
    trained_model = None

    # Data split, formatting
    dummy_X = observations[['lat', 'long']].as_matrix()
    dummy_y = observations['is_grand_larceny']

    # ZeroR Model
    dummy_clf = DummyClassifier(strategy='constant', constant=1)
    dummy_clf.fit(dummy_X, dummy_y)
    print('Dummy modle accuracy: {}'.format(dummy_clf.score(dummy_X, dummy_y)))

    # Keras model
    keras_x = observations[['lat', 'long', ]]


    logging.info('End model')
    lib.archive_dataset_schemas('model', locals(), globals())
    return observations, trained_model


def load(observations, trained_model):
    observations.to_csv('../data/output/train.csv', quoting=csv.QUOTE_ALL, index=False)
    pass


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)
    observations = extract()
    observations = transform(observations)
    observations, trained_model = model(observations)
    load(observations, trained_model)


# Main section
if __name__ == '__main__':
    main()
