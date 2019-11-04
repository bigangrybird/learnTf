#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals, absolute_import
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds


if __name__ == '__main__':
    train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])
    (train_data, validation_data), test_data = tfds.load(
        "imdb_reviews",
        split=(train_validation_split, tfds.Split.TEST),
        as_supervised=True
    )

    embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
    hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
    train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
    print(train_examples_batch)

    # model = tf.keras.Sequential([
    #     hub_layer,
    #     tf.keras.layers.Dense(16, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='sigmoid')
    # ])
    #
    # model.compile(optimizer='adam',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])
    #
    # model.fit(train_data.shuffle(10000).batch(512),
    #                 epochs=20,
    #                 validation_data=validation_data.batch(512),
    #                 verbose=1)