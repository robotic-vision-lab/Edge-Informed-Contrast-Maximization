import os

from dataloaders.reader_utils.hdf5_file_reader import HDF5FileReader
from dataloaders.reader_utils.numpy_file_reader import NumpyFileReader


class MVSECReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.h5_rdr = None
        self.np_rdr = None
        self.open_file = None
        self.close_file = None

        if os.path.splitext(self.file_path)[1].lower() in ['.hdf5', '.h5']:
            self._init_h5_reader()
        elif os.path.splitext(self.file_path)[1].lower() in ['.npz', '.npy']:
            self._init_np_reader()
        
    def _init_h5_reader(self):
        assert os.path.splitext(self.file_path)[1].lower() in ['.hdf5', '.h5'], (
            "Invalid hdf5 file extension"
        )
        self.h5_rdr = HDF5FileReader(self.file_path)
        self.open_file = self.open_h5_file
        self.close_file = self.close_h5_file
        self._reader = self.h5_rdr

    def _init_np_reader(self):
        assert os.path.splitext(self.file_path)[1].lower() in ['.npz', '.npy'], (
            "Invalid numpy file extension"
        )
        self.np_rdr = NumpyFileReader(self.file_path)
        self.open_file = self.open_np_file
        self.close_file = self.close_np_file
        self._reader = self.np_rdr
    
    def open_h5_file(self):
        return self.h5_rdr.open_file()
        
    def close_h5_file(self):
        self.h5_rdr.close_file()

    def open_np_file(self):
        return self.np_rdr.open_file()
    
    def close_np_file(self):
        self.np_rdr.close_file()

    def __enter__(self):
        return self.open_file()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close_file()

    def __repr__(self):
        return f"{type(self).__name__}(file_path='{self.file_path}')"

    def display_h5_datasets(self, disp=True):
        return self.h5_rdr.display_datasets(disp)
    
    def display_np_arrays(self, disp=True):
        return self.np_rdr.display_arrays(disp)

    def read_h5_dataset(self, dataset_name=None):
        if dataset_name is None:
            dataset_name = self.h5_rdr.display_datasets(disp=False)[0]

        return self.h5_rdr.read_dataset(dataset_name)
    
    def read_np_array(self, array_name=None):
        if array_name is None:
            array_name = self.np_rdr.display_arrays(disp=False)[0]

        return self.np_rdr.get_array(array_name)
