from pathlib import Path
from pdb import set_trace as st

import cv2 as cv
import numpy as np
from easydict import EasyDict as edict

from .mvsec_utils.mvsec_reader import MVSECReader


# # MVSEC dictionary
# MVSEC = edict()
# MVSEC.PATH = './datasets/MVSEC22' # relative to src (src will be in sys.path)
# MVSEC.SIZE = (260, 346)
# MVSEC.HEIGHT = 260
# MVSEC.WIDTH = 346
# MVSEC.SEQUENCES = {
#     'indoor_flying1': {'data_path': 'hdf5/indoor_flying/indoor_flying1_data.hdf5', 'gt_path': 'hdf5/indoor_flying/indoor_flying1_gt.hdf5', 'flow_gt_path': 'Flow GT/indoor_flying/indoor_flying1_gt_flow_dist.npz'},
#     'indoor_flying2': {'data_path': 'hdf5/indoor_flying/indoor_flying2_data.hdf5', 'gt_path': 'hdf5/indoor_flying/indoor_flying2_gt.hdf5', 'flow_gt_path': 'Flow GT/indoor_flying/indoor_flying2_gt_flow_dist.npz'},
#     'indoor_flying3': {'data_path': 'hdf5/indoor_flying/indoor_flying3_data.hdf5', 'gt_path': 'hdf5/indoor_flying/indoor_flying3_gt.hdf5', 'flow_gt_path': 'Flow GT/indoor_flying/indoor_flying3_gt_flow_dist.npz'},
#     'indoor_flying4': {'data_path': 'hdf5/indoor_flying/indoor_flying4_data.hdf5', 'gt_path': 'hdf5/indoor_flying/indoor_flying4_gt.hdf5', 'flow_gt_path': 'Flow GT/indoor_flying/indoor_flying4_gt_flow_dist.npz'},
#     'motorcycle1'   : {'data_path': 'motorcycle/motorcycle1_data.hdf5'},
#     'outdoor_day1'  : {'data_path': 'hdf5/outdoor_day/outdoor_day1_data.hdf5', 'gt_path': 'hdf5/outdoor_day/outdoor_day1_gt.hdf5', 'flow_gt_path': 'Flow GT/outdoor_day/outdoor_day1_gt_flow_dist.npz'},
#     'outdoor_day2'  : {'data_path': 'hdf5/outdoor_day/outdoor_day2_data.hdf5', 'gt_path': 'hdf5/outdoor_day/outdoor_day2_gt.hdf5', 'flow_gt_path': 'Flow GT/outdoor_day/outdoor_day2_gt_flow_dist.npz'},
#     'outdoor_night1': {'data_path': 'hdf5/outdoor_night/outdoor_night1_data.hdf5', 'gt_path': 'hdf5/outdoor_night/outdoor_night1_gt.hdf5', 'flow_gt_path': 'Flow GT/outdoor_night/outdoor_night1_gt_flow.npz'},
# }


class MVSECDataset:
    def __init__(self, root_dir, sequence_name):
        self.root_dir = Path(root_dir)
        self.sequence_name = sequence_name

        self.data_path = self.root_dir / f'hdf5/{sequence_name[:-1]}/{sequence_name}_data.hdf5'
        self.flow_gt_path = self.root_dir / f'Flow GT/{sequence_name[:-1]}/{sequence_name}_gt_flow_dist.npz'



class MVSECDataLoader:
    """
    This class loads MVSEC 
    
    + -----------------------------------------------------------------+
    | List of datasets inside hdf5 data file                           |
    | --------------------------------------                           |
    |     1. 'davis/left/events'                (N_l_evt, 4)           |
    |     2. 'davis/left/image_raw'             (N_l_img, 260, 346)    |
    |     3. 'davis/left/image_raw_event_inds'  (N_l_img, )            |
    |     4. 'davis/left/image_raw_ts'          (N_l_img, )            |
    |     5. 'davis/left/imu'                   (N_l_imu, 6)           |
    |     6. 'davis/left/imu_ts'                (N_l_imu, )            |
    |     7. 'davis/right/events'               (N_r_evt, 4)           |
    |     8. 'davis/right/image_raw'            (N_r_img, 260, 346)    |
    |     9. 'davis/right/image_raw_event_inds' (N_r_img, )            |
    |     10. 'davis/right/image_raw_ts'        (N_r_img, )            |
    |     11. 'davis/right/imu'                 (N_r_imu, 6)           |
    |     12. 'davis/right/imu_ts'              (N_r_imu, )            |
    |     13. 'velodyne/scans'                                         |
    |     14. 'velodyne/scans_ts'                                      |
    |                                                                  |
    | List of numpy arrays inside the .npz numpy file                  |
    | -----------------------------------------------                  |
    |     1. 'timestamps'                       (N_fgt, )              |
    |     2. 'y_flow_dist'                      (N_fgt, 260, 346)      |
    |     3. 'x_flow_dist'                      (N_fgt, 260, 346)      |
    |                                                                  |
    | Except for outdoor_night1, which has                             |
    |     1. 'x_flow_tensor'                    (N_fgt, 260,346)       |
    |     2. 'ps'                               (N_fgt, 3)             |
    |     3. 'qs'                               (N_fgt, 4)             |
    |     4. 'ts'                               (N_fgt, )              |
    |     5. 'y_flow_tensor'                    (N_fgt, 260,346)       |
    |     6. 'Vs'                               (N_fgt, 3)             |
    |     7. 'Omegas'                           (N_fgt, 3)             |
    + -----------------------------------------------------------------+
    """
    def __init__(self, 
                 root_dir, 
                 sequence_name, 
                 delta_idx=1, 
                 des_n_events=30_000, 
                 load_more_images=False, 
                 use_new_pruning_limits=False, 
                 prefer_latest_events=True):
        self.root_dir = Path(root_dir)
        self.sequence_name = sequence_name
        self.delta_idx = delta_idx
        self.des_n_events = des_n_events
        self.load_more_images = load_more_images
        self.use_new_pruning_limits = use_new_pruning_limits
        self.prefer_latest_events = prefer_latest_events
        self.n_event_deficiency = None

        self._LEFT_DATA_LOADED = False
        self._FLOW_GT_LOADED = False
        self._PRUNED = False

        self.dataset = MVSECDataset(self.root_dir, self.sequence_name)
        self.mvsec_h5_rdr = MVSECReader(self.dataset.data_path)
        self.mvsec_np_rdr = MVSECReader(self.dataset.flow_gt_path)


    def __getitem__(self, index):
        return self.get_sample_between_two_image_timestamps(index)
    

    def __len__(self):
        return len(self.l_image_raw_ts[self.delta_idx:])
    
  
    def get_ready(self):
        print(f'Loading ({self.sequence_name}) left data...')
        self.load_left_data()
        self.load_flow_gt()
        self.prune_data()
        print(f'Ready to load {self.sequence_name} datasamples.\n{"":-^80}')


    def load_left_data(self):
        """Loads all left data into memory
        """
        self.mvsec_h5_rdr.open_file()
        self.l_events_numpy = self.mvsec_h5_rdr.read_h5_dataset('davis/left/events')
        self.l_image_raw = self.mvsec_h5_rdr.read_h5_dataset('davis/left/image_raw')
        self.l_image_raw_event_inds = self.mvsec_h5_rdr.read_h5_dataset('davis/left/image_raw_event_inds')
        self.l_image_raw_ts = self.mvsec_h5_rdr.read_h5_dataset('davis/left/image_raw_ts')
        self.l_imu = self.mvsec_h5_rdr.read_h5_dataset('davis/left/imu')
        self.l_imu_ts = self.mvsec_h5_rdr.read_h5_dataset('davis/left/imu_ts')

        # crop events
        xs, ys, ts, ps = self.l_events_numpy.T
        xs = xs - 5
        ys = ys - 2
        ev_crop_mask = (xs >= 0) & (xs < 336) & (ys >= 0) & (ys < 256)
        xs = xs[ev_crop_mask]
        ys = ys[ev_crop_mask]
        ts = ts[ev_crop_mask]
        ps = ps[ev_crop_mask]
        # self.l_events = np.stack([xs, ys, ts, ps], axis=-1)
        self.l_events = {
            'x': xs.astype('int16'),
            'y': ys.astype('int16'),
            't': ts.astype('float64'),
            'p': ps.astype('bool'),
        }
        
        # crop images
        self.l_image_raw = self.l_image_raw[:, 2:-2, 5:-5]

        self.mvsec_h5_rdr.close_file()
        self._LEFT_DATA_LOADED = True


    def load_flow_gt(self):
        """Loads all flow groundtruth data into memory
        """
        self.mvsec_np_rdr.open_file()
        
        if 'outdoor_night1' not in self.mvsec_np_rdr.file_path.name.lower():
            self.gt_flow_ts = self.mvsec_np_rdr.read_np_array('timestamps')
            self.gt_x_flow_dist = self.mvsec_np_rdr.read_np_array('x_flow_dist')
            self.gt_y_flow_dist = self.mvsec_np_rdr.read_np_array('y_flow_dist')
        else:
            self.gt_flow_ts = self.mvsec_np_rdr.read_np_array('ts')
            self.gt_x_flow_dist = self.mvsec_np_rdr.read_np_array('x_flow_tensor')
            self.gt_y_flow_dist = self.mvsec_np_rdr.read_np_array('y_flow_tensor')

        # crop gt
        self.gt_x_flow_dist = self.gt_x_flow_dist[:, 2:-2, 5:-5]
        self.gt_y_flow_dist = self.gt_y_flow_dist[:, 2:-2, 5:-5]

        self.mvsec_np_rdr.close_file()
        self._FLOW_GT_LOADED = True


    def prune_data(self):
        """
        Prune data based on ground truth (qualitative) goodness.
        """
        assert self.sequence_name is not None
        assert self.sequence_name.lower() in [
            'indoor_flying1',
            'indoor_flying2',
            'indoor_flying3',
            'indoor_flying4',
            'outdoor_day1',
            'outdoor_day2',
            'outdoor_night1'
        ]
        assert self._LEFT_DATA_LOADED and self._FLOW_GT_LOADED, 'First, load data and flow gt'

        idx_start_gt = 0
        idx_stop_gt = None
        if "indoor_flying1" in self.sequence_name:
            idx_start_gt = 60  if self.use_new_pruning_limits else 60
            idx_stop_gt = 1340 if self.use_new_pruning_limits else 1340
        elif "indoor_flying2" in self.sequence_name:
            idx_start_gt = 150 if self.use_new_pruning_limits else 140
            idx_stop_gt = 1620 if self.use_new_pruning_limits else 1500
        elif "indoor_flying3" in self.sequence_name:
            idx_start_gt = 120 if self.use_new_pruning_limits else 100
            idx_stop_gt = 1825 if self.use_new_pruning_limits else 1711
        elif "indoor_flying4" in self.sequence_name:
            idx_start_gt = 104 if self.use_new_pruning_limits else 60
            idx_stop_gt = None if self.use_new_pruning_limits else 380
        elif "outdoor_day1" in self.sequence_name:
            idx_start_gt = 1 if self.use_new_pruning_limits else 0
            idx_stop_gt = 5040 if self.use_new_pruning_limits else 5020
        elif "outdoor_day2" in self.sequence_name:
            idx_start_gt = 20 if self.use_new_pruning_limits else 30
            idx_stop_gt = 12145 if self.use_new_pruning_limits else None
        elif "outdoor_night1" in self.sequence_name:
            idx_start_gt = 1
            idx_stop_gt = 5080

        # select ground truth slice
        self.gt_flow_ts = self.gt_flow_ts[idx_start_gt:idx_stop_gt]
        self.gt_x_flow_dist = self.gt_x_flow_dist[idx_start_gt:idx_stop_gt]
        self.gt_y_flow_dist = self.gt_y_flow_dist[idx_start_gt:idx_stop_gt]

        # update data
        idx_start_events = np.clip(np.searchsorted(self.l_events['t'], self.gt_flow_ts[0], side='left'),
                                   0, len(self.l_events['t']))
        idx_stop_events = np.clip(np.searchsorted(self.l_events['t'], self.gt_flow_ts[-1], side='right'),
                                  0, len(self.l_events['t']))
        idx_start_images = np.clip(np.searchsorted(self.l_image_raw_ts, self.gt_flow_ts[0], side='left'),
                                   0, len(self.l_image_raw))
        idx_stop_images = np.clip(np.searchsorted(self.l_image_raw_ts, self.gt_flow_ts[-1], side='right'),
                                  0, len(self.l_image_raw))
        idx_start_imu = np.clip(np.searchsorted(self.l_imu_ts, self.gt_flow_ts[0], side='left'),
                                0, len(self.l_imu))
        idx_stop_imu = np.clip(np.searchsorted(self.l_imu_ts, self.gt_flow_ts[-1], side='right'),
                               0, len(self.l_imu))

        # select event slice
        ev_slice = slice(
            min(idx_start_events, np.searchsorted(self.l_events['t'], self.l_image_raw_ts[idx_start_images])),
            max(idx_stop_events, np.searchsorted(self.l_events['t'], self.l_image_raw_ts[idx_stop_images]))
        )
        for tup_elem in ['x', 'y', 't', 'p']:
            self.l_events[tup_elem] = self.l_events[tup_elem][ev_slice]

        if self.sequence_name=='outdoor_day1':
            # filter events
            for tup_elem in ['x', 'y', 't', 'p']:
                self.l_events[tup_elem] = self.l_events[tup_elem][self.l_events['x'] < 190]
            
            
        
        # select images, image_ts, img_event_ind slice
        self.l_image_raw = self.l_image_raw[idx_start_images:idx_stop_images]
        self.l_image_raw_ts = self.l_image_raw_ts[idx_start_images:idx_stop_images]
        self.l_image_raw_event_inds = np.searchsorted(self.l_events['t'], self.l_image_raw_ts)
        
        # select imu slice
        self.l_imu = self.l_imu[idx_start_imu:idx_stop_imu]
        self.l_imu_ts = self.l_imu_ts[idx_start_imu:idx_stop_imu]

        # set the flag
        self._PRUNED = True


    def get_sample_between_two_image_timestamps(self, idx_img):
        """Return samples based on image timestamps
        """
        assert self._LEFT_DATA_LOADED and self._FLOW_GT_LOADED, 'First, load data and flow gt'
        # print(f'INFO: Pruned = {self.pruned}')

        # collect the start and end timestamps of gt based on index and delta
        t_img_start = self.l_image_raw_ts[idx_img]
        t_img_end = self.l_image_raw_ts[idx_img+self.delta_idx] 
        
        # select the two images spanning delta of idx
        if self.load_more_images:
            sampled_images = self.l_image_raw[idx_img: idx_img + self.delta_idx + 1]
            sampled_image_ts = self.l_image_raw_ts[idx_img: idx_img + self.delta_idx + 1]
        else:
            sampled_images = self.l_image_raw[[idx_img, idx_img + self.delta_idx]]
            sampled_image_ts = np.array([t_img_start, t_img_end])
        
        # Note:
        # -----
        # + delta would mean that the images and events in the 'next' delta must predict the current gt_flow
        # - delta would mean that the images and events in the 'prev' delta must predict the current gt_flow
        # 
        # MVSEC dataset is setup in the '+ delta' fashion

        # select events within delta of idx i.e., start and end gt timestamps
        idx_evt_start  = np.searchsorted(self.l_events['t'], t_img_start, side='left')
        idx_evt_end = np.searchsorted(self.l_events['t'], t_img_end, side='right')
        orig_n_events = (idx_evt_end - idx_evt_start)
        if self.des_n_events is not None:
            # make sure we have desired num of events (corner cases not handled)
            self.n_event_deficiency = self.des_n_events - (idx_evt_end - idx_evt_start) 
            if self.n_event_deficiency > 0:
                idx_evt_start -= np.ceil(self.n_event_deficiency/2).astype(int)
                idx_evt_end += np.floor(self.n_event_deficiency/2).astype(int)
                idx_evt_start = max(0, idx_evt_start)
                idx_evt_end = min(idx_evt_end, len(self.l_events['t']))
            elif self.n_event_deficiency < 0:
                # TODO
                #  if there are more events then keep the boundary events 
                # and remove randomly from the middle section
                if self.prefer_latest_events:
                    idx_evt_start = idx_evt_end - self.des_n_events
                else:
                    idx_evt_end = idx_evt_start + self.des_n_events
        
        sampled_events = {}
        for tup_elem in ['x', 'y', 't', 'p']:
            sampled_events[tup_elem] = self.l_events[tup_elem][idx_evt_start:idx_evt_end]

        # select imu data within delta of idx i.e., start and end gt timestamps
        idx_imu_start = np.searchsorted(self.l_imu_ts, t_img_start, side='left')
        idx_imu_end = np.searchsorted(self.l_imu_ts, t_img_end, side='right')
        sampled_imu = self.l_imu[idx_imu_start:idx_imu_end]
        sampled_imu_ts = self.l_imu_ts[idx_imu_start:idx_imu_end]

        # estimate gt_flow at idx_img
        u_est, v_est = self.estimate_gt_flow(t_img_start, t_img_end)
        # u_est, v_est = self.estimate_gt_flow(sampled_events[2,0], sampled_events[2,-1])
        estimated_gt_flow = np.stack([u_est, v_est], axis=-1)
        estimated_gt_flow_ts = sampled_image_ts[[0, -1]]

        return {
            'events': sampled_events,
            'images': sampled_images,
            'image_ts': sampled_image_ts,
            'flow_gt': estimated_gt_flow,
            'eval_ts': estimated_gt_flow_ts,
            'imu': sampled_imu,
            'imu_ts': sampled_imu_ts,
            'n_event_deficiency': self.n_event_deficiency,
            'orig_n_events': orig_n_events,
        }



    def estimate_gt_flow(self, t_start, t_end):
        """Adapted and improved from https://github.com/daniilidis-group/EV-FlowNet/blob/master/src/eval_utils.py#L95
        """        
        idx_gt = np.searchsorted(self.gt_flow_ts, t_start, side='right') - 1
        gt_dt = self.gt_flow_ts[idx_gt + 1] - self.gt_flow_ts[idx_gt]
        x_flow = np.squeeze(self.gt_x_flow_dist[idx_gt, ...])
        y_flow = np.squeeze(self.gt_y_flow_dist[idx_gt, ...])
        dt = t_end - t_start
        pre_dt = self.gt_flow_ts[idx_gt + 1] - t_start 

        # --------------------------------------------------------------------------------
        # dt = 1 setting (to be interpreted as dt = 1 img_dt)
        # No need to propagate if the desired dt is shorter than the time between gt timestamps.
        if gt_dt >= dt and pre_dt >= dt:
            return (x_flow * dt / gt_dt, 
                    y_flow * dt / gt_dt)

        # --------------------------------------------------------------------------------
        # dt > 1 setting (to be interpreted as dt > 1 img_dt)  
        #
        #                             
        # indices         :  0   1 ... idx_gt     
        #                    v   v       v        dt          gt_dt
        #                    |   |       | |<---------->|    |<->|
        # gt_ts           :  |   |  ...  |   |   |   |   |   |   |   |   |
        # (t_start, t_end):                |..--------..|
        #                                  ^            ^
        #                               t_start       t_end
        # 
        # In the illustration above, we can observe that to obtain ground truth flow at t_start, 
        # representing the flow between t_start and t_end, we need to propagate the flow
        # through three sections:
        #   (1) left dots (pre_dt), 
        #   (2) middle dashes (mid_dt), and 
        #   (3) right dots (end_dt)
        # Note, (1) and (3) involve flow propagation in partial gt_dt amounts, 
        # while (2) involves propagating by whole multiples of gt_dt amounts.

        # generate coordinate matrices
        x_coords, y_coords = np.meshgrid(np.arange(x_flow.shape[1]), 
                                         np.arange(x_flow.shape[0]), 
                                         indexing='xy')
        
        x_coords = x_coords.astype(np.float32) # (H, W)
        y_coords = y_coords.astype(np.float32) # (H, W)

        # make a copy for computing total shift later on
        orig_x_coords = np.copy(x_coords) # (H, W)
        orig_y_coords = np.copy(y_coords) # (H, W)
        
        # mask keeps track of the points that leave the image, and zeros out the flow afterwards.
        x_mask = np.ones(x_coords.shape, dtype=bool) # (H, W)
        y_mask = np.ones(y_coords.shape, dtype=bool) # (H, W)

        # propagate flow for preceeding partial pre_dt
        scale_factor = pre_dt / gt_dt        
        self._prop_flow(x_flow, y_flow, x_coords, y_coords, x_mask, y_mask, scale_factor=scale_factor)

        # propagate flow for middle runs of n*gt_dt (=mid_dt)
        idx_gt += 1
        while self.gt_flow_ts[idx_gt + 1] < t_end:
            x_flow = np.squeeze(self.gt_x_flow_dist[idx_gt, ...]) # (H, W)
            y_flow = np.squeeze(self.gt_y_flow_dist[idx_gt, ...]) # (H, W)
            # scale_factor is gt_dt/gt_dt, i.e., 1.0
            self._prop_flow(x_flow, y_flow, x_coords, y_coords, x_mask, y_mask, scale_factor=1.0)

            idx_gt += 1
        
        # propagate flow for end partial dt
        end_dt = t_end - self.gt_flow_ts[idx_gt]
        end_gt_dt = self.gt_flow_ts[idx_gt + 1] - self.gt_flow_ts[idx_gt]

        x_flow = np.squeeze(self.gt_x_flow_dist[idx_gt, ...]) # (H, W)
        y_flow = np.squeeze(self.gt_y_flow_dist[idx_gt, ...]) # (H, W)
        scale_factor = end_dt / end_gt_dt
        
        self._prop_flow(x_flow, y_flow, x_coords, y_coords, x_mask, y_mask, scale_factor)

        # compute the shift/displacement (flow) in coords
        x_shift = x_coords - orig_x_coords
        y_shift = y_coords - orig_y_coords

        # mask out/ zero out wherever flow_._interp was zero (mask was False)
        x_shift[~x_mask] = 0
        y_shift[~y_mask] = 0

        return x_shift, y_shift


    def _prop_flow(self, x_flow, y_flow, x_coords, y_coords, x_mask, y_mask, scale_factor=1.0):
        """
        Code obtained from https://github.com/daniilidis-group/EV-FlowNet/blob/master/src/eval_utils.py#L51C5-L51C14
        Updates numpy arrays x_coords, y_coords, x_mask, y_mask, based on the flow x_flow and y_flow.
        """
        flow_x_interp = cv.remap(x_flow,
                                x_coords,
                                y_coords,
                                cv.INTER_NEAREST)
        
        flow_y_interp = cv.remap(y_flow,
                                x_coords,
                                y_coords,
                                cv.INTER_NEAREST)

        x_mask[flow_x_interp == 0] = False
        y_mask[flow_y_interp == 0] = False
            
        x_coords += flow_x_interp * scale_factor
        y_coords += flow_y_interp * scale_factor

        # no need to return coords and mask, since numpy arrays are mutable
        return 



    def get_sample_between_two_gt_timestamps(self, idx_gt):
        """Return samples based on gt timestamps
        """
        assert self._LEFT_DATA_LOADED and self._FLOW_GT_LOADED, 'First, load data and flow gt'
        print(f'INFO: Pruned = {self.pruned}')

        # collect the start and end timestamps of gt based on index and delta
        t_gt_start = self.gt_flow_ts[idx_gt]
        t_gt_end = self.gt_flow_ts[idx_gt+self.delta_idx] 

        # Note:
        # -----
        # + delta would mean that the images and events in the 'next' delta must predict the current gt_flow
        # - delta would mean that the images and events in the 'prev' delta must predict the current gt_flow
        # 
        # MVSEC dataset is setup in the '+ delta' fashion

        # select events within delta of idx i.e., start and end gt timestamps
        idx_evt_start  = np.searchsorted(self.l_events['t'], t_gt_start, side='left')
        idx_evt_end = np.searchsorted(self.l_events['t'], t_gt_end, side='right')
        sampled_events = {}
        for tup_elem in ['x', 'y', 't', 'p']:
            sampled_events[tup_elem] = self.l_events[tup_elem][idx_evt_start:idx_evt_end]

        # select images within delta of idx i.e., start and end gt timestamps
        idx_img_start  = np.searchsorted(self.l_image_raw_ts, t_gt_start, side='left')
        idx_img_end = np.searchsorted(self.l_image_raw_ts, t_gt_end, side='right')
        sampled_images = self.l_image_raw[idx_img_start:idx_img_end]
        sampled_image_ts = self.l_image_raw_ts[idx_img_start:idx_img_end]

        # select imu data within delta of idx i.e., start and end gt timestamps
        idx_imu_start = np.searchsorted(self.l_imu_ts, t_gt_start, side='left')
        idx_imu_end = np.searchsorted(self.l_imu_ts, t_gt_end, side='right')
        sampled_imu = self.l_imu[idx_evt_start:idx_imu_end]
        sampled_imu_ts = self.l_imu_ts[idx_evt_start:idx_imu_end]

        # select gt within delta of idx i.e., start and end gt timestamps
        sampled_gt_flow = np.stack([self.gt_x_flow_dist[idx_gt], self.gt_y_flow_dist[idx_gt]], axis=0)
        sampled_gt_flow_ts = np.array([t_gt_start, t_gt_end])
        
        return {
            'events': sampled_events,
            'images': sampled_images,
            'image_ts': sampled_image_ts,
            'flow_gt': sampled_gt_flow,
            'eval_ts': sampled_gt_flow_ts,
            'imu': sampled_imu,
            'imu_ts': sampled_imu_ts          
        }


    def index_to_time(self, idx):
        return self.l_events['t'][idx]


    def time_to_index(self, t):
        return np.searchsorted(self.l_events['t'], t) - 1


    @property
    def left_data(self):
        return (
            self.l_events,
            self.l_image_raw,
            self.l_image_raw_event_inds,
            self.l_image_raw_ts,
            self.l_imu,
            self.l_imu_ts
        )


    @property
    def left_flow_data(self):
        return (
            self.l_events,
            self.l_image_raw,
            self.l_image_raw_event_inds,
            self.l_image_raw_ts,
        )


    @property
    def flow_gt(self):
        return (
            self.gt_flow_ts,
            self.gt_x_flow_dist,
            self.gt_y_flow_dist
        )
        
    
    @property
    def pruned(self):
        return self._PRUNED

