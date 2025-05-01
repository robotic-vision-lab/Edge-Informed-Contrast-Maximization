import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from pathlib import Path


ECD_HEIGHT = 176
ECD_WIDTH  = 240

class ECDDataset:
    def __init__(self, root_dir, sequence_name):
        self.root_dir = Path(root_dir)
        
        self.images_dir = self.root_dir / f'{sequence_name}/images'
        self.events_path = self.root_dir / f'{sequence_name}/events.txt'
        self.calibration_path = self.root_dir / f'{sequence_name}/calibration.txt'
        self.gt_path = self.root_dir / f'{sequence_name}/groundtruth.txt'
        self.image_ts_path = self.root_dir / f'{sequence_name}/images.txt'
        


class ECDDataLoader:
    def __init__(self, root_dir, sequence_name, des_n_events=50_000, delta_idx=1, prefer_latest_events=True):
        self.root_dir = Path(root_dir)
        self.sequence_name = sequence_name
        self.des_n_events = des_n_events
        self.delta_idx = delta_idx
        self.prefer_latest_events = prefer_latest_events

        self.height = ECD_HEIGHT
        self.width = ECD_WIDTH
        self.sensor_size = (self.height, self.width)
        
        self.dataset = ECDDataset(root_dir=self.root_dir, sequence_name=self.sequence_name)


    def get_ready(self):
        print(f'Loading {self.sequence_name} data...')
        self.load_events()
        self.load_images()
        self.precompute_eval_event_indices()
        self.precompute_eval_image_indices()
        print(f'Ready to load {self.sequence_name} datasamples.\n{"":-^80}')


    def load_events(self):
        events_npz = np.loadtxt(self.dataset.events_path, delimiter=' ', skiprows=0, dtype=np.float64)
        self.events = {
            't': events_npz[:, 0],
            'x': events_npz[:, 1].astype('int16'),
            'y': events_npz[:, 2].astype('int16'),
            'p': events_npz[:, 3].astype('bool'),
        }

        
        self.events['y'] = self.events['y'] - 2
        ev_crop_mask = (self.events['y'] >= 0) & (self.events['y'] < 176)
        self.events['x'] = self.events['x'][ev_crop_mask]
        self.events['y'] = self.events['y'][ev_crop_mask]
        self.events['t'] = self.events['t'][ev_crop_mask]
        self.events['p'] = self.events['p'][ev_crop_mask]


    def load_images(self):
        self.image_ts = []
        with open(self.dataset.image_ts_path, '+r') as f:
            for line in f.readlines():
                self.image_ts.append(float(line.split(' ')[0].strip()))

        self.image_ts = np.array(self.image_ts)

        self.eval_ts = np.array([self.image_ts[:-self.delta_idx], self.image_ts[self.delta_idx:]])

        self.image_paths = sorted([str(p) for p in self.dataset.images_dir.iterdir() if str(p).endswith('.png')])


    def precompute_eval_event_indices(self):
        self.eval_event_start_idxs = np.searchsorted(self.events['t'], self.eval_ts[0, :], side='left')
        self.eval_event_end_idxs = np.searchsorted(self.events['t'], self.eval_ts[1, :], side='left')


    def precompute_eval_image_indices(self):
        self.eval_image_start_idxs = np.searchsorted(self.image_ts, self.eval_ts[0, :], side='left')
        self.eval_image_end_idxs = np.searchsorted(self.image_ts, self.eval_ts[1, :], side='left')


    def get_sample(self, eval_idx):
        # prepare sample images
        idx_img_start, idx_img_end = self.eval_image_start_idxs[eval_idx], self.eval_image_end_idxs[eval_idx]
        sampled_images_paths = self.image_paths[idx_img_start : idx_img_end+1]
        sampled_images = np.array([cv.imread(im_path, cv.IMREAD_GRAYSCALE) for im_path in sampled_images_paths])
        
        # crop images
        sampled_images = sampled_images[:, 2:-2, :]

        # prepare sample events
        idx_evt_start, idx_evt_end = self.eval_event_start_idxs[eval_idx], self.eval_event_end_idxs[eval_idx]
        orig_n_events = (idx_evt_end - idx_evt_start)
        if self.des_n_events is not None:
            # make sure we have desired num of events (corner cases not handled)
            self.n_event_deficiency = self.des_n_events - (idx_evt_end - idx_evt_start) 
            if self.n_event_deficiency > 0:
                idx_evt_start -= np.ceil(self.n_event_deficiency/2).astype(int)
                idx_evt_end += np.floor(self.n_event_deficiency/2).astype(int)
                idx_evt_start = max(0, idx_evt_start)
                idx_evt_end = min(idx_evt_end, len(self.events['x']))
            elif self.n_event_deficiency < 0:
                # TODO
                #  if there are more events then keep the boundary events 
                # and remove randomly from the middle section
                if self.prefer_latest_events:
                    idx_evt_start = idx_evt_end - self.des_n_events
                else:
                    idx_evt_end = idx_evt_start + self.des_n_events

        sampled_events = {
            'x': self.events['x'][idx_evt_start:idx_evt_end],
            'y': self.events['y'][idx_evt_start:idx_evt_end],
            't': self.events['t'][idx_evt_start:idx_evt_end],
            'p': self.events['p'][idx_evt_start:idx_evt_end],
        }


        return {
            'events': sampled_events,
            'images': sampled_images,
            'image_ts': self.image_ts[idx_img_start : idx_img_end+1],
            'eval_ts': self.eval_ts[:, eval_idx],
            'n_event_deficiency': self.n_event_deficiency,
            'orig_n_events': orig_n_events,
        }


    def __getitem__(self, idx):
        return self.get_sample(idx)
  
    
    def __len__(self):
        return len(self.eval_ts)
    



