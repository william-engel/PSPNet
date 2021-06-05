import tensorflow as tf
import os 
import numpy as np

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if tf.is_tensor(value):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(img_fpath, target_size = None):

    # .../leftImg8bit/.../berlin_000000_000019_leftImg8bit.png
    # .../gtFine/.../berlin_000000_000019_gtFine_labelIds.png
    mask_fpath = tf.strings.regex_replace(img_fpath, 'leftImg8bit', 'gtFine', replace_global=False)
    mask_fpath = tf.strings.regex_replace(mask_fpath, 'leftImg8bit', 'gtFine_labelIds', replace_global=False)

    # load image
    image = tf.io.read_file(img_fpath)
    image = tf.image.decode_png(image, channels=3)

    # load mask
    mask = tf.io.read_file(mask_fpath)
    mask = tf.image.decode_png(mask, channels=1)

    # resize
    if target_size is not None:
        image = tf.image.resize(image, target_size, method = 'nearest')
        mask = tf.image.resize(mask, target_size, method = 'nearest')

    # image format
    image_format = b'png'
    mask_format = b'png'

    # encode
    image_encoded = tf.io.encode_png(image)
    mask_encoded = tf.io.encode_png(mask)

    filename = os.path.basename(img_fpath).encode('utf-8')

    height, width, channels = image.get_shape()

    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.

    feature = {
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/channels': _int64_feature(channels),
        'image/filename': _bytes_feature(filename),
        'image/encoded': _bytes_feature(image_encoded),
        'image/format': _bytes_feature(image_format),
        'image/segmentation/class/encoded': _bytes_feature(mask_encoded),
        'image/segmentation/class/format':_bytes_feature(mask_format)   
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def create_tfrecord_shards(img_fpaths, num_shards, type, target_size):
    # type 'train', 'val', 'test'
    
    tfrecords_fpath_placeholder = '/content/cityscapes_dataset/tfrecords/{}/cityscapes_{}_{}_of_{}.records'

    img_fpaths_shards = np.array_split(img_fpaths, num_shards)

    for index, img_fpaths_shard in enumerate(img_fpaths_shards):
        tfrecords_fpath = tfrecords_fpath_placeholder.format(type, type, index+1, num_shards)
        writer = tf.io.TFRecordWriter(tfrecords_fpath)
        for img_fpath in img_fpaths_shard:
            example = serialize_example(img_fpath, target_size)
            writer.write(example)
            writer.close()

# Create a dictionary describing the features.
image_feature_description = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/channels': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/segmentation/class/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/segmentation/class/format':tf.io.FixedLenFeature([], tf.string)  
}

def _parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.io.decode_png(example['image/encoded'])
    mask  = tf.io.decode_png(example['image/segmentation/class/encoded'])
    return image, mask
