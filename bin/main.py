#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import csv
import logging

import pandas

import lib
def extract():
    observations = pandas.read_csv(lib.get_conf('train_path'))
    lib.archive_dataset_schemas('extract', locals(), globals())
    return observations

def transform(observations):

    observations.columns = map(lambda x: '_'.join(x.lower().split()), observations.columns)
    observations['lat'] = observations['location_1'].apply(lambda x: eval(x)[0])
    observations['long'] = observations['location_1'].apply(lambda x: eval(x)[1])
    lib.archive_dataset_schemas('transform', locals(), globals())
    return observations

def model(observations):
    trained_model = None
    return observations, trained_model

def load(observations, trained_model):
    observations.to_csv('../data/output/train.csv', quoting = csv.QUOTE_ALL, index=False)
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
    load(observations, trained_model )
    pass


# Main section
if __name__ == '__main__':
    main()
