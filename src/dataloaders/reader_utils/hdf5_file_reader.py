import h5py


class HDF5FileReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.h5_file = None

    def open_file(self):
        try:
            self.h5_file = h5py.File(self.file_path, 'r')
            return self
        except Exception as e:
            print(f'Error opening HDF5 file: {str(e)}')
            return None
        
    def close_file(self):
        if self.h5_file:
            self.h5_file.close()

    def __enter__(self):
        return self.open_file()
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close_file()

    def __repr__(self):
        return f"{type(self).__name__}(file_path='{self.file_path}')"

    def display_datasets(self, disp=True):
        _datasets = []
        def _func(name, obj):
            if isinstance(obj, h5py.Dataset):
                _datasets.append(name)

        self.h5_file.visititems(_func)

        if disp:
            _ = [print(f"{i}. '{ds}'") for i, ds in enumerate(_datasets, start=1)]
        
        return _datasets

    def read_dataset(self, dataset_name):
        if not self.h5_file and not self.open_file(): return None

        try:
            print(f"Reading hdf5 dataset '{dataset_name}'")
            dataset = self.h5_file[dataset_name]
            data = dataset[:] # load everything from dataset into memory
            return data
        except Exception as e:
            print(f"Error reading dataset '{dataset_name}': {str(e)}")
            return None
