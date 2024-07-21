import tensorflow as tf
from pathlib import Path
from absl.logging import info, warning, fatal, error, debug

def parse_path(p, dataroot=None):
    if dataroot is None:
        dataroot = Path.cwd()
    if p[0] == '/':
        _p = Path('/')
        p = '.' + p
    else:
        _p = Path(dataroot)

    filenames = list(pp.as_posix() for pp in _p.glob(p))
    debug('----------------------------------------------')
    debug(f'globed from {p}')
    debug(f'found {len(filenames)} files')
    return filenames


def load_classifier_dataset(filenames, 
                            repeat=False, 
                            shuffle=False, 
                            batchsize=32, 
                            random_flip=False,
                            dataroot=None,
                           ):
    raw_ds = tf.data.TFRecordDataset(parse_path(filenames, dataroot))
    feature_description = {
        'width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }

    def _parse_example(x):
        return tf.io.parse_single_example(x, feature_description)

    #get width and height from first example
    for eg in raw_ds:
        row = tf.io.parse_single_example(eg, feature_description)
        width = int(row['width'])
        height = int(row['height'])
        break

    def _decode_image(x):
        img = tf.io.decode_raw(x['image'], tf.float32)
        img = tf.reshape(img, (width, height, 1))
        label = x['label']

        return img, label

    def _random_lr_flip(x, y):
        return tf.image.random_flip_left_right(x), y

    ds = raw_ds.map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(_decode_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if random_flip:
        ds.map(_random_lr_flip)

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(batchsize*4)
    else:
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    ds = ds.batch(batchsize)

    return ds



