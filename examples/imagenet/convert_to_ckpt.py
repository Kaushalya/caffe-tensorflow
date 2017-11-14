#!/usr/bin/env python
'''Validates a converted ImageNet model against the ILSVRC12 validation set.'''

import argparse
import numpy as np
import tensorflow as tf
import os.path as osp

import models
import dataset

def load_model(name):
    '''Creates and returns an instance of the model given its class name.
    The created model has a single placeholder node for feeding images.
    '''
    # Find the model class from its name
    all_models = models.get_models()
    lut = {model.__name__: model for model in all_models}
    if name not in lut:
        print('Invalid model index. Options are:')
        # Display a list of valid model names
        for model in all_models:
            print('\t* {}'.format(model.__name__))
        return None
    NetClass = lut[name]

    # Create a placeholder for the input image
    spec = models.get_data_spec(model_class=NetClass)
    data_node = tf.placeholder(tf.float32,
                               shape=(None, spec.crop_size, spec.crop_size, spec.channels))

    # Construct and return the model
    return NetClass({'data': data_node})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Path to the converted model parameters (.npy)')
    parser.add_argument('--model', default='AlexNet', help='The name of the model to evaluate')
    parser.add_argument('--ckpt_path', help='Where to save the converted checkpoint file')
    args = parser.parse_args()

    # Load the network
    net = load_model(args.model)
    if net is None:
        exit(-1)

    saver = tf.train.Saver()
    data_spec = models.get_data_spec(model_instance=net)

    with tf.Session() as sess:
        net.load(data_path=args.model_path, session=sess)
        saver.save(sess, args.ckpt_path)



if __name__ == '__main__':
    main()
