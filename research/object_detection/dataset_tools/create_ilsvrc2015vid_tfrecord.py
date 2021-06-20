'''
Usage:
Run from models/research directory.

# You should modify the path below on line 191, to point to the training for only VID image files.
vid_filepath = 'object_detection/tfrecord/VID_trainimg_small.txt'

# You should modify the path below on line 229, to point to the training for VID and DET image files.
det_filepath = 'object_detection/tfrecord/DET_train_30classes.txt'
vid_filepath = 'object_detection/tfrecord/VID_train_15frames.txt'

# You should modify the path below on line 290, to point to the validation for VID image files.
vid_filepath = 'object_detection/tfrecord/VID_val_videos.txt'

# For VID only train
python object_detection/dataset_tools/create_ilsvrc2015vid_tfrecord.py \
--output_path object_detection/tfrecord/temp/ilsvrc2015_train \
--set train --data_dir /local/scratch/a/wang4495/ILSVRC2015 \
--num_shards 10

# For DET+VID train
python object_detection/dataset_tools/create_ilsvrc2015vid_tfrecord.py \
--output_path object_detection/tfrecord/temp/ilsvrc2015_train_vid_det \
--set train_with_det --data_dir /local/scratch/a/wang4495/ILSVRC2015 \
--num_shards 50

# For Val set
python object_detection/dataset_tools/create_ilsvrc2015vid_tfrecord.py \
--output_path /local/scratch/a/wang4495/work/efficientdet/automl/efficientdet/dataset/tfrecord/ilsvrc2015_val \
--set val --data_dir /local/scratch/a/wang4495/ILSVRC2015 \
--num_shards 75
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import contextlib2

from lxml import etree
import PIL.Image
import tensorflow.compat.v1 as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from object_detection.dataset_tools import tf_record_creation_util


flags = tf.app.flags
flags.DEFINE_string(
    'data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'object_detection/tfrecord/ilsvrc2015.pbtxt',
                    'Path to label map proto')
tf.flags.DEFINE_integer('num_shards', 50, 'Number of TFRecord shards')
FLAGS = flags.FLAGS

SETS = ['train', 'train_with_det', 'val']

class_set = [
    'n02691156',
    'n02419796',
    'n02131653',
    'n02834778',
    'n01503061',
    'n02924116',
    'n02958343',
    'n02402425',
    'n02084071',
    'n02121808',
    'n02503517',
    'n02118333',
    'n02510455',
    'n02342885',
    'n02374451',
    'n02129165',
    'n01674464',
    'n02484322',
    'n03790512',
    'n02324045',
    'n02509815',
    'n02411705',
    'n01726692',
    'n02355227',
    'n02129604',
    'n04468005',
    'n01662784',
    'n04530566',
    'n02062744',
    'n02391049'
]

class UniqueId(object):
  """Class to get the unique {image/ann}_id each time calling the functions."""

  def __init__(self):
    self.image_id = 0
    self.ann_id = 0

  def get_image_id(self):
    self.image_id += 1
    return self.image_id

  def get_ann_id(self):
    self.ann_id += 1
    return self.ann_id

def dict_to_tf_example(data,
                       full_path,
                       dataset_directory,
                       label_map_dict,
                       unique_id):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      dataset_directory: Path to root directory holding PASCAL dataset
      label_map_dict: A map from string label names to integers ids.
      unique_id: UniqueId object to get the unique {image/ann}_id for the image
      and the annotations.
      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
      image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image data.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """

    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    image_id = unique_id.get_image_id()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []

    if 'object' in data:
        for obj in data['object']:
            if obj['name'] in class_set:
                xmin.append(float(obj['bndbox']['xmin']) / width)
                ymin.append(float(obj['bndbox']['ymin']) / height)
                xmax.append(float(obj['bndbox']['xmax']) / width)
                ymax.append(float(obj['bndbox']['ymax']) / height)
                classes_text.append(obj['name'].encode('utf8'))
                classes.append(label_map_dict[obj['name']])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            str(image_id).encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example

# TODO:
# Add reading from txt file
# Add sharding

def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    '''
    data_dir should be root dir
    /mnt/d/Datasets/ILSVRC2015/
        - Annotations
            - DET
            - VID
        - Data
            - DET
            - VID
    '''
    data_dir = FLAGS.data_dir

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    if FLAGS.set == 'train':
        vid_filepath = 'object_detection/tfrecord/VID_trainimg_small.txt'

        logging.info('Starting tfrecord generation for ILSVRC2015 train for VID.')

        vid_anno_dir = os.path.join(data_dir, 'Annotations', 'VID')
        vid_img_dir = os.path.join(data_dir, 'Data', 'VID')

        with tf.gfile.GFile(vid_filepath) as fid:
            lines = fid.readlines()
        examples_list = [line.strip()[9:-5] for line in lines]

        unique_id = UniqueId()

        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
                tf_record_close_stack,
                FLAGS.output_path,
                FLAGS.num_shards
            )

            for idx, example in enumerate(examples_list):
                if idx % 1000 == 0:
                    logging.info('On image %d of %d', idx, len(examples_list))
                
                anno_path = os.path.join(vid_anno_dir, example + '.xml')
                img_path = os.path.join(vid_img_dir, example + '.JPEG')

                with tf.gfile.GFile(anno_path, 'r') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
                tf_example = dict_to_tf_example(data, img_path, FLAGS.data_dir, label_map_dict, unique_id)

                shard_idx = idx % FLAGS.num_shards
                output_tfrecords[shard_idx].write(tf_example.SerializeToString())

    elif FLAGS.set == 'train_with_det':
        det_filepath = 'object_detection/tfrecord/DET_train_30classes.txt'
        vid_filepath = 'object_detection/tfrecord/VID_train_15frames.txt'

        det_anno_dir = os.path.join(data_dir, 'Annotations', 'DET')
        det_img_dir = os.path.join(data_dir, 'Data', 'DET')

        vid_anno_dir = os.path.join(data_dir, 'Annotations', 'VID')
        vid_img_dir = os.path.join(data_dir, 'Data', 'VID')

        logging.info('Starting tfrecord generation for ILSVRC2015 train with VID + DET.')

        # For DET
        with tf.gfile.GFile(det_filepath) as fid:
            lines = fid.readlines()
        det_len = len(lines)
        examples_list = [line.strip().split(' ')[0] for line in lines]

        # For VID
        with tf.gfile.GFile(vid_filepath) as fid:
            lines = fid.readlines()
        # Add behind the DET file
        examples_list.extend([line.strip().split(' ')[0] + '/%06d' % int(line.strip().split(' ')[2]) for line in lines])
        
        del lines
        
        unique_id = UniqueId()

        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
                tf_record_close_stack,
                FLAGS.output_path,
                FLAGS.num_shards
            )
            
            for idx, example in enumerate(examples_list):
                if idx % 1000 == 0:
                    logging.info('On image %d of %d', idx, len(examples_list))
                if idx < det_len:  # For DET Dataset
                    anno_path = os.path.join(det_anno_dir, example + '.xml')
                    img_path = os.path.join(det_img_dir, example + '.JPEG')

                    with tf.gfile.GFile(anno_path, 'r') as fid:
                        xml_str = fid.read()
                    
                    xml = etree.fromstring(xml_str)
                    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
                    tf_example = dict_to_tf_example(data, img_path, FLAGS.data_dir, label_map_dict, unique_id)

                else:  # For VID Dataset
                    anno_path = os.path.join(vid_anno_dir, example + '.xml')
                    img_path = os.path.join(vid_img_dir, example + '.JPEG')

                    with tf.gfile.GFile(anno_path, 'r') as fid:
                        xml_str = fid.read()
                    xml = etree.fromstring(xml_str)
                    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
                    tf_example = dict_to_tf_example(data, img_path, FLAGS.data_dir, label_map_dict, unique_id)

                shard_idx = idx % FLAGS.num_shards
                output_tfrecords[shard_idx].write(tf_example.SerializeToString())

    elif FLAGS.set == 'val':
        # For validation set.
        vid_filepath = 'object_detection/tfrecord/VID_val_videos.txt'

        vid_anno_dir = os.path.join(data_dir, 'Annotations', 'VID')
        vid_img_dir = os.path.join(data_dir, 'Data', 'VID')        

        logging.info('Starting tfrecord generation for ILSVRC2015 val.')

        with tf.gfile.GFile(vid_filepath) as fid:
            lines = fid.readlines()
        examples_list = [line.strip().split(' ')[0] + '/%06d' % int(line.strip().split(' ')[2]) for line in lines]

        unique_id = UniqueId()
        
        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
                tf_record_close_stack,
                FLAGS.output_path,
                FLAGS.num_shards
            )

            for idx, example in enumerate(examples_list):
                if idx % 1000 == 0:
                    logging.info('On image %d of %d', idx, len(examples_list))
                
                anno_path = os.path.join(vid_anno_dir, example + '.xml')
                img_path = os.path.join(vid_img_dir, example + '.JPEG')

                with tf.gfile.GFile(anno_path, 'r') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
                tf_example = dict_to_tf_example(data, img_path, FLAGS.data_dir, label_map_dict, unique_id)

                shard_idx = idx % FLAGS.num_shards
                output_tfrecords[shard_idx].write(tf_example.SerializeToString())
    else:
        logging.warning('Set not recognized. Please check the FLAGS.set .')

if __name__ == '__main__':
    tf.app.run()
