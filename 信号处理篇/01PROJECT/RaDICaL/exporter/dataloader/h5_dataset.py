import h5py
import tensorflow as tf
from absl.logging import error, warning, info, debug, fatal
import numpy as np

class H5UnalignedDatasetSaver(object):
    def __init__(self, filename, stream_names, 
                 initial_size=10,
                 size_increment=10,
                 include_timestamps=True,
                ):
        self.filename = filename
        num_streams = len(stream_names)
        self.f = h5py.File(filename, mode='w')
        self._insert_idx = [0] * num_streams
        self._curr_size = [initial_size] * num_streams
        self._initial_size = initial_size
        self._inc_size = size_increment
        self.stream_names = stream_names
        self.stream_shapes = [None] * num_streams
        self.include_timestamps = include_timestamps

    def _create_dataset(self, data, stream_idx):
        d, dt = data
        dt = np.array(dt)
        name = self.stream_names[stream_idx]
        self.f.create_dataset(name,
                              [self._initial_size]+list(d.shape),
                              dtype=d.dtype,
                              maxshape=([None] + list(d.shape)))
        self.f.create_dataset(f'{name}_timestamp',
                              [self._initial_size]+[1],
                              dtype=dt.dtype,
                              maxshape=([None] + [1]))


    def push_frame(self, data, timestamp, stream_idx):
        name = self.stream_names[stream_idx]
        if not self.f.get(name):
            self._create_dataset((data, timestamp), stream_idx)
        self.f[name][self._insert_idx[stream_idx]] = data
        self.f[f'{name}_timestamp'][self._insert_idx[stream_idx]] = timestamp
        self._insert_idx[stream_idx] += 1
        
        self.increase_size_if_full(stream_idx)


    def increase_size_if_full(self, stream_idx):
        if self._curr_size[stream_idx] == self._insert_idx[stream_idx]:
            name = self.stream_names[stream_idx]
            new_size = self.f[name].shape[0] + self._inc_size
            shape = list(self.f[name].shape[1:])
            self.f[f'{name}_timestamp'].resize([new_size] + [1])
            self.f[name].resize([new_size] + shape)
            self._curr_size[stream_idx] = new_size

    def _trim(self):
        for idx, name in enumerate(self.stream_names):
            shape = list(self.f[name].shape[1:])
            self.f[name].resize([self._insert_idx[idx]] + shape)
            self.f[f'{name}_timestamp'].resize([self._insert_idx[idx]] + [1])

    def __del__(self):
        self._trim()



class H5DatasetSaver(object):
    """docstring for DatasetSaver"""
    def __init__(self, filename, stream_names, 
                 initial_size=10,
                 size_increment=10,
                 include_timestamps=False,
                ):
        super(H5DatasetSaver, self).__init__()
        self.filename = filename
        self.f = h5py.File(filename, mode='w')
        self._insert_idx = 0
        self._curr_size = initial_size
        self._initial_size = initial_size
        self._inc_size = size_increment
        self.stream_names = stream_names
        self.stream_shapes = []
        self.include_timestamps = include_timestamps


        self.created = False

    def _lazy_create(self, data):
        for name, d in zip(self.stream_names, data):
            if self.include_timestamps:
                d, dt = d
                dt = np.array(dt)
                self.f.create_dataset(f'{name}_timestamp',
                                      [self._initial_size]+[1],
                                      dtype=dt.dtype,
                                      maxshape=([None] + [1]))
            self.stream_shapes.append(list(d.shape))
            self.f.create_dataset(name, 
                                  [self._initial_size]+list(d.shape), 
                                  dtype=d.dtype,
                                  maxshape=([None] + list(d.shape)))

                                      
        self.created = True

    def push(self, data):
        if not self.created:
            self._lazy_create(data)
            debug(self.shapes)
        for name, d in zip(self.stream_names, data):
            if self.include_timestamps:
                d, dt = d
                dt = np.array(dt)
                self.f[f'{name}_timestamp'][self._insert_idx] = dt
            self.f[name][self._insert_idx] = d

        self._insert_idx += 1

        if self._insert_idx >= self._curr_size:
            debug(self.shapes)
            self.resize(self._curr_size + self._inc_size)
            debug(self.shapes)

    def resize(self, new_size):
        for name, shape in zip(self.stream_names, self.stream_shapes):
            if self.include_timestamps:
                self.f[f'{name}_timestamp'].resize([new_size] + [1])
            self.f[name].resize([new_size] + shape)
            self._curr_size = new_size

    @property
    def shapes(self):
        return {n: self.f[n].shape for n in self.stream_names}

    def __del__(self):
        #remove extra frames
        self.resize(self._insert_idx)

class H5DatasetLoader(object):
    """A thin wrapper around h5py to provide convenience functions for training"""
    def __init__(self, filenames):
        super(H5DatasetLoader, self).__init__()
        self.filenames = filenames
        if isinstance(self.filenames, list):
            raise NotImplementedError
        else:
            self.h5_file = h5py.File(self.filenames, 'r')
        self.streams_available = list(self.h5_file.keys())

    def __len__(self):
        return len(self.h5_file[self.streams_available[0]])

    def __getitem__(self, stream):
        return self.h5_file[stream]

    @property
    def filename(self):
        return self.filenames
        
    def get_tf_dataset(self,
                       streams=['radar', 'rgb', 'depth'],
                       shuffle=False,
                       repeat=False,
                       batchsize=16,
                       preprocess_chain=None,
                       prefetch=2,
                       flatten_single=False,
                      ):
        debug("Tensorflow Dataset creation")

        out_shapes = tuple([ 
            tf.TensorShape(list(self.h5_file[s].shape[1:])) for s in streams
        ])
        out_types = tuple([self.h5_file[s].dtype for s in streams])

        def _gen():
            for i in range(len(self.h5_file[streams[0]])):
                yield tuple(self.h5_file[s][i] for s in streams)

        _dataset = tf.data.Dataset.from_generator(
            _gen,
            output_types = out_types,
            output_shapes = out_shapes,
        )

        if shuffle:
            debug("  Outputs of dataset will be shuffled")
            _dataset = _dataset.shuffle(batchsize * 4)

        if repeat:
            debug(f'  Dataset will be repeated {repeat} files')
            _dataset = _dataset.repeat(repeat)

        if preprocess_chain is not None:
            for op in preprocess_chain:
                _dataset = _dataset.map(op)

        if flatten_single:
            assert(len(streams) == 1)
            debug("  Flattening shapes for single stream inference")
            debug(_dataset)
            _dataset = _dataset.map(lambda x: x)
            debug(_dataset)

        _dataset = _dataset.batch(batchsize)


        if prefetch:
            _dataset = _dataset.prefetch(prefetch)

        return _dataset


