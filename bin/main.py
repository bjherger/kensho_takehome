#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import csv
import logging

import numpy
import pandas
from sklearn.cross_validation import train_test_split
from sklearn.dummy import DummyClassifier

import lib
import models


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)

    # Extract data from upstream sources
    observations = extract()

    # Transform data for model use
    observations, X, y, label_encoder = transform(observations)

    # Tran model
    observations, X, y, label_encoder, trained_model = model(observations, X, y, label_encoder)

    # Export data, model, and other assets for external / future use
    load(observations, X, y, label_encoder, trained_model)


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

    regressors = ['occurrence_day', 'occurrence_year', 'compstat_month', 'compstat_day', 'compstat_year', 'lat', 'long']
    response_var = 'response'

    X = observations[regressors].as_matrix().astype(numpy.float32)
    y = numpy.array(observations[response_var].tolist()).astype(numpy.float32)

    logging.info('End transform')
    return observations, X, y, label_encoder


def model(observations, X, y, label_encoder):
    logging.info('Beginning model')

    # # Data split, formatting
    # dummy_X = observations[['lat', 'long']].as_matrix()
    # dummy_y = observations['is_grand_larceny']
    #
    # # ZeroR Model
    # dummy_clf = DummyClassifier(strategy='constant', constant=1)
    # dummy_clf.fit(dummy_X, dummy_y)
    # print('Dummy modle accuracy: {}'.format(dummy_clf.score(dummy_X, dummy_y)))

    # Keras model

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_test_mask = numpy.random.random(size=len(observations.index))
    num_train = sum(train_test_mask < .8)
    num_validate = sum(train_test_mask >= .8)
    logging.info('Proceeding w/ {} train observations, and {} test observations'.format(num_train, num_validate))

    ff_model = models.gen_stupid_ff_network(X.shape[1], y.shape[1])

    ff_model.fit(X_train, y_train, batch_size=1024, epochs=4, validation_data=(X_test, y_test))

    # Add predictions to data set
    preds = ff_model.predict(X)
    observations['max_probability'] = map(max, preds)
    observations['prediction_index'] = map(lambda x: numpy.argmax(x))
    observations['modeling_prediction'] = map(lambda x: lib.prop_to_label(x, label_encoder), preds)
    trained_model = ff_model
    logging.info('End model')
    lib.archive_dataset_schemas('model', locals(), globals())
    return observations, X, y, label_encoder, trained_model


def load(observations, X, y, label_encoder, trained_model):
    logging.info('Begin load')
    observations.to_csv('../data/output/train.csv', quoting=csv.QUOTE_ALL, index=False)
    logging.info('End load')
    pass


# Main section
if __name__ == '__main__':
    main()
