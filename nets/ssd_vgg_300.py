# Copyright 2016 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Definition of 300 VGG-based SSD network.
"""

import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

import tf_extended as tfe
from nets import custom_layers
from nets import ssd_common

slim = tf.contrib.slim


# =========================================================================== #
# SSD class definition.
# =========================================================================== #
SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'])


class SSDNet(object):
    """
    Implementation of the SSD VGG-based 300 network.

    The default features layers with 300x300 image input are:
        conv4 ==> 38 x 38
        conv7 ==> 19 x 19
        conv8 ==> 10 x 10
        conv9 ==> 5 x 5
        conv10 ==> 3 x 3
        conv11 ==> 1 x 1
    The default image size used to train this network is 300x300.
    """
    default_params = SSDParams(
        img_shape=(300, 300),
        num_classes=21,
        no_annotation_label=21,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        anchor_size_bounds=[0.15, 0.90],
        anchor_sizes=[(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)],
        anchor_ratios=[[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5],
                       [2, .5]],
        anchor_steps=[8, 16, 32, 64, 100, 300],
        anchor_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

    def __init__(self, params=None):
        """
        Init the SSD net with some parameters. Use the default ones if none provided.
        :param params:
        """
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet.default_params

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_300_vgg'):
        """
        SSD network definition.
        """
        r = ssd_net(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Try to update feature shapes
        if update_feat_shapes:
            shapes = ssd_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r    # Contains: predictions, localisations, logits, end_points

    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        """
        Network arg_scope.
        """
        return ssd_arg_scope(weight_decay, data_format=data_format)

    def arg_scope_caffe(self):
        pass

    # ======================================================================= #
    def update_feature_shapes(self):
        pass

    def anchors(self, img_shape, dtype=np.float32):
        """
        Compute the default anchor boxes, given an image shape.
        """
        return ssd_anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)


def ssd_net(inputs,
            num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,  # a list
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_300_vgg'):
    """
    SSD net definition.
    """
    end_points = {}
    with tf.variable_scope(scope, 'ssd_300_vgg', [inputs], reuse=reuse):
        # Original VGG-16 blocks.
        # Block 1.
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['block3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        end_points['block4'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['block5'] = net
        net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')

        # Additional SSD blocks.
        # Block 6: let's dilate the hell out of it!
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
        end_points['block6'] = net
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
        # Block 7: 1x1 conv. Because the fuck.
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
        end_points['block7'] = net
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)

        # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
        end_point = 'block8'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block9'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block10'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block11'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):     # block4_box
                p, l = ssd_multibox_layer(end_points[layer],
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i])
                predictions.append(prediction_fn(p))
                logits.append(p)
                localisations.append(l)

    return predictions, localisations, logits, end_points


def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    """
    Defines the VGG arg scope.
    :param weight_decay: The l2 regularization coefficient.
    :param data_format:
    :return: An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([custom_layers.pad2d,
                                 custom_layers.l2_normalization,
                                 custom_layers.channel_to_last],
                                data_format=data_format) as sc:
                return sc


# =========================================================================== #
# Functional definition of VGG-based SSD 300.
# =========================================================================== #
def tensor_shape(x, rank=3):
    """
    Returns the dimensions of a tensor.
    Args:
        image: A N-D Tensor of shape.
    Returns:
        A list of dimensions. Dimensions that are statically known are python
        integers, otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def ssd_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """
    Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    if normalization > 0:
        net = custom_layers.l2_normalization(net, scaling=True)

    # Number of anchors.
    num_anchors = len(sizes) + len(ratios)

    # Location.
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None, scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1] + [num_anchors, 4])

    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None, scope='conv_cls')
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1] + [num_anchors, num_classes])
    return cls_pred, loc_pred


# =========================================================================== #
# SSD tools...
# =========================================================================== #
def ssd_size_bounds_to_values(size_bounds,
                              n_feat_layers,
                              img_shape=(300, 300)):
    pass


def ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """
    Try to obtain the feature shape from the prediciton layers.
    The latter can be either a Tensor or Numpy ndarray.
    :param predictions:
    :param default_shapes:
    :return: List of feature shapes. Default values if predicitons shape not fully determined.
    """
    feat_shapes = []
    for l in predictions:
        # Get the shape, from either a np array or a tensor
        if isinstance(l, np.ndarray):
            shape = l.shape
        else:
            shape = l.get_shape().as_list()
        shape = shape[1:4]
        # Problem: undetermined shape...
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes


# =========================================================================== #
# For anchors.
# =========================================================================== #
def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """
    Compute SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative width and height.
    :return: Relative x and y grids, and height and width.
    """
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)

    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w


def ssd_anchors_all_layers(img_shape,
                           layers_shapes,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """
    Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, layer_shape in enumerate(layers_shapes):
        anchor_bboxes = ssd_anchor_one_layer(img_shape,
                                             layer_shape,  # i.e. feat shape
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset,
                                             dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors

