# Copyright 2015 Paul Balanca. All Rights Reserved.
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
# 2018-08-14 -- Qi

"""
Converts Pascal VOC data to TFRecords file format with Example protos.
"""

import os
import sys
import random
import tensorflow as tf
import xml.etree.ElementTree as ET

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature

from datasets.pascalvoc_common import VOC_LABELS


# Original dataset organisation.
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200


def _process_image(directory, name):
    """
    Process an image and corresponding annotation file.
    :param directory:
    :param name:
    :return:
    """
    # Read the image file.
    filename = directory + DIRECTORY_IMAGES + name + '.jpg'
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    # Read the XML annotation file.
    filename = directory + DIRECTORY_ANNOTATIONS + name + '.xml'
    tree = ET.parse(filename)
    root = tree.getroot()

    # XML: Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]

    # XML: annotations
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)

        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ))

    return image_data, shape, bboxes, labels, labels_text, difficult, truncated


def _convert_to_example(image_data, shape, bboxes, labels, labels_text,
                        difficult, truncated):
    """
    Build an Example proto for an image example.

    :param image_data:
    :param shape:
    :param bboxes:
    :param labels:
    :param labels_text:
    :param difficult:
    :param truncated:
    :return: Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    # This three lines convert tuple (which contains a serial of bounding boxes into
    # list format.
    # e.g. [(1, 2, 3, 4), (5, 6, 7, 8)] ==> a list, each element is a bounding box.
    # will be convert to ymin=[1, 5], xmin=[2, 6], ymax=[3, 7] and xmax=[4, 8]
    for b in bboxes:
        assert len(b) == 4
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/object/bbox/difficult': int64_feature(difficult),
        'image/object/bbox/truncated': int64_feature(truncated),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)
    }))
    return example


def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    """
    Loads data from image and annotation files and add them to TFRecord.
    :param dataset_dir: Dataset directory.
    :param name: Image name to add to the TFRecord.
    :param tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = _process_image(dataset_dir, name)
    example = _convert_to_example(image_data, shape, bboxes, labels, labels_text, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, name='voc_train', shuffling=False):
    """
    Runs the convertion operation.

    :param dataset_dir: The dataset directory where the dataset is stored.
    :param output_dir: Output directory.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    # Dataset filenames and shuffling.
    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    filenames = sorted(os.listdir(path))

    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(filenames):
        # Open new TFRecord file
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                img_name = filename[:-4]
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1
    # Finally, write the labels file.
    print('\nFinished converting the Pascal VOC dataset.')

