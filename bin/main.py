#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import logging


def extract():
    observations = None
    return observations

def transform(observations):
    return observations

def model(observations):
    trained_model = None
    return observations, trained_model

def load(observations, trained_model):
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
