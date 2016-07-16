
# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras.engine import Layer
import keras.objectives


class Loss(Layer):
    def __init__(self, loss, **kwargs):
        self.loss = loss
        self.loss_fn = keras.objectives.get(loss)
        super(Loss, self).__init__(**kwargs)

    def compute_loss(self, input, output, input_mask=None, output_mask=None):
        return output

    def get_output_shape_for(self, input_shape):
        x_shape, y_shape = input_shape
        return (None, 1)

    def call(self, input, mask=None):
        if type(input) is not list:
            raise Exception("Loss layer takes two input tensors. Only got one {}.".format(input))
        if len(input) != 2:
            raise Exception("Loss layer takes two input tensors. Only got {}.".format(len(input)))

        x, y = input
        return self.loss_fn(x, y)

    def get_config(self):
        config = {'loss': self.loss}
        base_config = super(Loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
