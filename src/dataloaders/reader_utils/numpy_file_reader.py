import numpy as np


class NumpyFileReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.np_file = None

    def open_file(self):
        try:
            self.np_file = np.load(self.file_path)
            return self
        except Exception as e:
            print(f"Error reading numpy file: {str(e)}")
            return None
        
    def close_file(self):
        if self.np_file:
            self.np_file.close()

    def __enter__(self):
        return self.open_file()
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close_file()
        
    def __repr__(self):
        return f"{type(self).__name__}(file_path='{self.file_path}')"

    def display_arrays(self, disp=True):
        arrays = [array_name for array_name in self.np_file.keys()]
        if disp:
            _ = [print(f"{i}. '{arr_name}'") for i, arr_name in enumerate(arrays, start=1)]

        return arrays
    
    def get_array(self, array_name=None):
        if self.np_file is None and not self.open_file(): return None

        if array_name in self.display_arrays(disp=False):
            print(f"Reading numpy array '{array_name}'")
            return self.np_file[array_name]
        else:
            print(f"Array {array_name} not found in numpy file")
            return None