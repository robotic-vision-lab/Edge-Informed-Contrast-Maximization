from pathlib import Path
from pdb import set_trace as st

import cv2 as cv
import imageio.v2 as imageio
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

from dataloaders.dsec_utils.euclidean_transform import Transform
from dataloaders.reader_utils.hdf5_file_reader import HDF5FileReader



class DSECTestDataset:
    def __init__(self, root_dir, sequence_name, extended):
        self.root_dir = root_dir
        self.extended = extended

        self.events_h5_path = root_dir / f'Test/test_events/{sequence_name}/events/left/events.h5'
        self.rectify_map_h5_path = root_dir / f'Test/test_events/{sequence_name}/events/left/rectify_map.h5'
        self.calib_cam_to_cam_yml_path = root_dir / f'Test/test_calibration/{sequence_name}/calibration/cam_to_cam.yaml'

        self.l_images_timestamps_path = root_dir / f'Test/test_images/{sequence_name}/images/timestamps.txt'
        self.l_images_exposure_timestamps_path = root_dir / f'Test/test_images/{sequence_name}/images/left/exposure_timestamps.txt'
        self.l_images_dir = root_dir / f'Test/test_images/{sequence_name}/images/left/rectified'
        if self.extended:
            self.test_forward_optical_flow_timestamps_path = root_dir / f'Evaluation/test_forward_optical_flow_timestamps/{sequence_name}_.csv'
        else:
            self.test_forward_optical_flow_timestamps_path = root_dir / f'Evaluation/test_forward_optical_flow_timestamps/{sequence_name}.csv'


class DSECTrainDataset:
    def __init__(self, root_dir, sequence_name):
        self.root_dir = root_dir

        self.events_h5_path = root_dir / f'Train/train_events/{sequence_name}/events/left/events.h5'
        self.rectify_map_h5_path = root_dir / f'Train/train_events/{sequence_name}/events/left/rectify_map.h5'
        self.calib_cam_to_cam_yml_path = root_dir / f'Train/train_calibration/{sequence_name}/calibration/cam_to_cam.yaml'

        self.l_images_timestamps_path = root_dir / f'Train/train_images/{sequence_name}/images/timestamps.txt'
        self.l_images_exposure_timestamps_path = root_dir / f'Train/train_images/{sequence_name}/images/left/exposure_timestamps.txt'
        self.l_images_dir = root_dir / f'Train/train_images/{sequence_name}/images/left/rectified'

        self.flow_gt_forward_timestamps_path = root_dir / f'Train/train_optical_flow/{sequence_name}/flow/forward_timestamps.txt'
        self.flow_gt_forward_dir = root_dir / f'Train/train_optical_flow/{sequence_name}/flow/forward'


class DSECDataLoader:
    def __init__(self, 
                 root_dir, 
                 sequence_name, 
                 des_n_events=1_500_000, 
                 data_split='test', 
                 extended=False, 
                 prefer_latest_events=True):
        self.root_dir = Path(root_dir)
        self.sequence_name = sequence_name
        self.des_n_events = des_n_events
        self.data_split = data_split
        self.extended = extended
        self.prefer_latest_events = prefer_latest_events

        # set event sensor size
        self.height = 480
        self.width = 640
        self.sensor_size = (480, 640)

        self.dataset = (DSECTestDataset(self.root_dir, self.sequence_name, extended) if data_split == 'test' 
                        else DSECTrainDataset(self.root_dir, self.sequence_name))

        self._LEFT_DATA_LOADED = False
        self._FLOW_GT_LOADED = False
        self._EVENTS_RECTIFIED = False
        self._IMAGE_MAPPING_CONSTRUCTED = False
        self._EVENT_REC_MAP_CONSTRUCTED = False


    def get_ready(self):
        print(f'Loading ({self.sequence_name}) left data...')
        self.load_left_data()
        self.load_flow_gt()
        self.rectify_events()
        self.construct_mapping_for_image()
        self.construct_event_rectify_map_from_calibration()
        self.precompute_eval_event_indices()
        self.precompute_eval_image_indices()
        print(f'\nReady to load {self.sequence_name} datasamples.\n{"":-^80}')


    def load_left_data(self):
        # load from events.h5
        with HDF5FileReader(self.dataset.events_h5_path) as h5_rdr:
            p = (h5_rdr.read_dataset('events/p')).astype('bool')
            t = h5_rdr.read_dataset('events/t') # microseconds
            x = (h5_rdr.read_dataset('events/x')).astype('int16')
            y = (h5_rdr.read_dataset('events/y')).astype('int16')
            
            self.ms_to_idx = h5_rdr.read_dataset('ms_to_idx')
            self.t_offset = h5_rdr.h5_file['t_offset'][()]
            self.l_events = {'x': x, 'y': y, 't': t, 'p': p}

        # load from rectify_map.h5
        with HDF5FileReader(self.dataset.rectify_map_h5_path) as h5_rdr:
            self.rectify_map = h5_rdr.read_dataset('rectify_map')

        # load cam_to_cam.yaml
        with open(self.dataset.calib_cam_to_cam_yml_path) as f:
            self.cam_to_cam = yaml.safe_load(f)

        # load timestamps.txt, exposure_timestamps.txt, and /rectified image directory
        self.l_image_ts_us = np.loadtxt(self.dataset.l_images_timestamps_path, skiprows=0, dtype='int64')
        self.l_image_exp_ts_us = np.loadtxt(self.dataset.l_images_exposure_timestamps_path, delimiter=',', skiprows=1, dtype='int64')
        self.l_image_dir = self.dataset.l_images_dir
        self.l_image_paths = sorted([str(p) for p in self.l_image_dir.iterdir() if str(p).endswith('.png')])

        # load ground truth optical flow dir
        if self.data_split == 'train':
            self.flow_gt_dir = self.dataset.flow_gt_forward_dir
            self.flow_gt_paths = sorted([str(p) for p in self.flow_gt_dir.iterdir() if str(p).endswith('.png')])
            self.eval_ts_us = np.loadtxt(self.dataset.flow_gt_forward_timestamps_path, delimiter=',', skiprows=1, dtype='int64')

        # eval timestamps
        if self.data_split == 'test':
            self.eval_ts_us = np.loadtxt(self.dataset.test_forward_optical_flow_timestamps_path, delimiter=',', skiprows=1, dtype='int64')

        self._LEFT_DATA_LOADED = True


    def load_flow_gt(self):
        print('Loading flow ground truth... ')
        if self.data_split.lower() == 'test':
            print('Test split. No GT to load.')
            return None
        elif self.data_split.lower() == 'train':
            self.flow_gt_ts = np.loadtxt(self.dataset.flow_gt_forward_timestamps_path, delimiter=',', skiprows=1, dtype='int64')
            self.flow_gt_dir = self.dataset.flow_gt_forward_dir
        
        self._FLOW_GT_LOADED = True


    def rectify_events(self):
        assert self.rectify_map.shape == (self.height, self.width, 2), self.rectify_map.shape
        assert self.l_events['x'].max() < self.width
        assert self.l_events['y'].max() < self.height
        
        print('Rectifying events... ')
        rect_event_coords = self.rectify_map[self.l_events['y'], self.l_events['x']]
        rec_x, rec_y = rect_event_coords.T
        rec_x = np.round(rec_x).astype('int16')
        rec_y = np.round(rec_y).astype('int16')
        self.l_events['x'], self.l_events['y']  = rec_x, rec_y
        print('\bDone.')

        print('Filtering out-of-sensor rectified events - computing in-sensor mask... ')
        rec_x_mask = np.logical_and(rec_x >= 0, rec_x < self.width)
        rec_y_mask = np.logical_and(rec_y >= 0, rec_y < self.height)
        rect_event_mask = np.logical_and(rec_x_mask, rec_y_mask)
        print('\bDone.')

        # keep only events that are inside size
        print('Filtering out-of-sensor rectified events - applying in-sensor mask... ')
        for tup_elem in ['x', 'y', 't', 'p']:
            self.l_events[tup_elem] = self.l_events[tup_elem][rect_event_mask]
        print('\bDone.')

        self._EVENTS_RECTIFIED = True
        return None


    def precompute_eval_event_indices(self):
        print('Pre-computing eval event indices for efficiency... ')
        self.eval_event_start_idxs = np.searchsorted(self.l_events['t'], self.eval_ts_us[:, 0] - self.t_offset, side='left')
        self.eval_event_end_idxs = np.searchsorted(self.l_events['t'], self.eval_ts_us[:, 1] - self.t_offset, side='left')
        print('\bDone.')
    

    def precompute_eval_image_indices(self):
        print('Pre-computing eval image indices for efficiency... ')
        self.eval_image_start_idxs = np.searchsorted(self.l_image_ts_us, self.eval_ts_us[:, 0], side='left')
        self.eval_image_end_idxs = np.searchsorted(self.l_image_ts_us, self.eval_ts_us[:, 1], side='left')
        print('\bDone.')


    def construct_mapping_for_image(self):
        print('Constructing mapping for image... ')
        K_r0 = np.eye(3)
        K_r0[[0, 1, 0, 1], [0, 1, 2, 2]] = self.cam_to_cam['intrinsics']['camRect0']['camera_matrix']
        K_r1 = np.eye(3)
        K_r1[[0, 1, 0, 1], [0, 1, 2, 2]] = self.cam_to_cam['intrinsics']['camRect1']['camera_matrix']

        R_r0_0 = Rot.from_matrix(np.array(self.cam_to_cam['extrinsics']['R_rect0']))
        R_r1_1 = Rot.from_matrix(np.array(self.cam_to_cam['extrinsics']['R_rect1']))

        T_r0_0 = Transform.from_rotation(R_r0_0)
        T_r1_1 = Transform.from_rotation(R_r1_1)
        T_1_0 = Transform.from_transform_matrix(np.array(self.cam_to_cam['extrinsics']['T_10']))

        T_r1_r0 = T_r1_1 @ T_1_0 @ T_r0_0.inverse()
        R_r1_r0_matrix = T_r1_r0.R().as_matrix()
        P_r1_r0 = K_r1 @ R_r1_r0_matrix @ np.linalg.inv(K_r0)
        
        # coords: ht, wd, 2
        coords = np.stack(np.meshgrid(np.arange(self.width), np.arange(self.height)), axis=-1)
        # coords_hom: ht, wd, 3
        coords_hom = np.concatenate((coords, np.ones((self.height, self.width, 1))), axis=-1)
        # mapping: ht, wd, 3
        mapping = (P_r1_r0 @ coords_hom[..., None]).squeeze()
        # mapping: ht, wd, 2
        mapping = (mapping/mapping[..., -1][..., None])[..., :2]
        self.mapping = mapping.astype('float32')
        print('\bDone.')

        self._IMAGE_MAPPING_CONSTRUCTED = True

        return self.mapping


    def construct_event_rectify_map_from_calibration(self):
        print('Constructing rectify map for events... ')
        K_0 = np.eye(3)
        K_0[[0, 1, 0, 1], [0, 1, 2, 2]] = self.cam_to_cam['intrinsics']['cam0']['camera_matrix']
        K_r0 = np.eye(3)
        K_r0[[0, 1, 0, 1], [0, 1, 2, 2]] = self.cam_to_cam['intrinsics']['camRect0']['camera_matrix']
        dist_coeffs = np.array(self.cam_to_cam['intrinsics']['cam0']['distortion_coeffs'])
        R_r0_0 = np.array(self.cam_to_cam['extrinsics']['R_rect0'])

        coords = np.stack(np.meshgrid(np.arange(self.width), np.arange(self.height))).reshape((2, -1)).astype("float32")
        term_criteria = (cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 100, 0.001)
        points = cv.undistortPointsIter(coords, K_0, dist_coeffs, R_r0_0, K_r0, criteria=term_criteria)
        inv_map = points.reshape((self.height, self.width, 2))
        print('\bDone.')
        self.event_rect_map = inv_map

        self._EVENT_REC_MAP_CONSTRUCTED = True

        return self.event_rect_map


    def map_image_to_rect_event(self, img):
        return cv.remap(img, self.mapping, None, interpolation=cv.INTER_CUBIC)
    

    @staticmethod
    def flow_16bit_to_float(flow_16bit: np.ndarray):
        assert flow_16bit.dtype == np.uint16
        assert flow_16bit.ndim == 3
        h, w, c = flow_16bit.shape
        assert c == 3

        valid2D = flow_16bit[..., 2] == 1
        assert valid2D.shape == (h, w)
        assert np.all(flow_16bit[~valid2D, -1] == 0)
        valid_map = np.where(valid2D)

        # to actually compute something useful:
        flow_16bit = flow_16bit.astype('float')

        flow_map = np.zeros((h, w, 2))
        flow_map[valid_map[0], valid_map[1], 0] = (flow_16bit[valid_map[0], valid_map[1], 0] - 2 ** 15) / 128
        flow_map[valid_map[0], valid_map[1], 1] = (flow_16bit[valid_map[0], valid_map[1], 1] - 2 ** 15) / 128
        return flow_map, valid2D


    @staticmethod
    def load_flow(flowfile: Path):
        assert flowfile.exists()
        assert flowfile.suffix == '.png'
        flow_16bit = imageio.imread(str(flowfile), format='PNG-FI')
        flow, valid2D = DSECDataLoader.flow_16bit_to_float(flow_16bit)
        return flow, valid2D


    def index_to_time(self, event_idx):
        return self.l_events['t'][event_idx]
    

    def time_to_index(self, event_t_us):
        return np.searchsorted(self.l_events['t'], event_t_us) - 1


    def get_sample(self, eval_idx):
        # prepare sample images
        idx_img_start, idx_img_end = self.eval_image_start_idxs[eval_idx], self.eval_image_end_idxs[eval_idx]
        sampled_images_paths = self.l_image_paths[idx_img_start : idx_img_end+1]
        sampled_images = [
            self.map_image_to_rect_event(cv.imread(im_path, cv.IMREAD_GRAYSCALE))
            for im_path in sampled_images_paths
        ]

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
                idx_evt_end = min(idx_evt_end, len(self.l_events['x']))
            elif self.n_event_deficiency < 0:
                # TODO
                #  if there are more events then keep the boundary events 
                # and remove randomly from the middle section
                if self.prefer_latest_events:
                    idx_evt_start = idx_evt_end - self.des_n_events
                else:
                    idx_evt_end = idx_evt_start + self.des_n_events

        sampled_events = {
            'x': self.l_events['x'][idx_evt_start:idx_evt_end],
            'y': self.l_events['y'][idx_evt_start:idx_evt_end],
            't': self.l_events['t'][idx_evt_start:idx_evt_end] + self.t_offset,
            'p': self.l_events['p'][idx_evt_start:idx_evt_end],
        }


        if self.data_split == 'train':
            # prepare gt
            flow_gt, valid2D = DSECDataLoader.load_flow(Path(self.flow_gt_paths[eval_idx]))

        ret = None
        if self.data_split == 'test':
            ret = {
                'events': sampled_events,
                'images': sampled_images,
                'image_ts': self.l_image_ts_us[idx_img_start : idx_img_end+1],
                'eval_ts_us': self.eval_ts_us[eval_idx, :2],
                'file_idx': self.eval_ts_us[eval_idx, 2],
                'n_event_deficiency': self.n_event_deficiency,
                'orig_n_events': orig_n_events,
            }
        elif self.data_split == 'train':
            ret = {
                'events': sampled_events,
                'images': sampled_images,
                'image_ts': self.l_image_ts_us[idx_img_start : idx_img_end+1],
                'eval_ts_us': self.eval_ts_us[eval_idx, :2],
                'file_idx': self.eval_ts_us[eval_idx, 2],
                'flow_gt': flow_gt,
                'valid2D': valid2D,
                'n_event_deficiency': self.n_event_deficiency,
                'orig_n_events': orig_n_events,
            }

        return ret
            

    @property
    def is_left_data_loaded(self):
        return self._LEFT_DATA_LOADED


    @property
    def is_flow_gt_loaded(self):
        return self._FLOW_GT_LOADED


    def __getitem__(self, idx):
        return self.get_sample(idx)


    def __len__(self):
        return len(self.eval_ts_us)