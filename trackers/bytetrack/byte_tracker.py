import cv2
import numpy as np
import warnings
import copy
from scipy.signal import find_peaks
from easydict import EasyDict as edict

from trackers.bytetrack import matching
from trackers.bytetrack.basetrack import BaseTrack, TrackState
from trackers.bytetrack.kalman_filter import KalmanFilter


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
    Returns:
       y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    y = np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, cls, obj_pos=None):
        # wait activate
        self.xywh = tlwh
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.obj_pos = obj_pos

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        # self.cls = cls
        self.xywh = new_track.xywh
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, track_thresh=0.45, match_thresh=0.8, track_buffer=25, frame_rate=30, roi_points=None):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.track_buffer = track_buffer

        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.det_thresh = track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        self.roi = edict()
        if roi_points:
            bottom_right = roi_points["br"]  # .strip().split()
            top_left = roi_points["tl"]  # .strip().split()
            top_right = [bottom_right[0], top_left[1]]
            bottom_left = [top_left[0], bottom_right[1]]
            self.roi.list_of_points = [bottom_right, top_right, top_left, bottom_left]
            self.list_of_points = np.array([self.roi.list_of_points], np.int32)
        else:
            self.roi.list_of_points = []

    # Filter detections based on FOV
    def filter_detections(self, xywh, clss):
        position_list = []
        picker_location = []
        for index, (det, cls) in enumerate(zip(xywh, clss)):
        
            position_list.append("FOV")
           
        return position_list, picker_location

    # change jay : to detect change in direction to be used as and when required based on use case
    def detect_sudden_change(self, idx, xy):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if idx not in self.object_trajectories.keys():
                return False
            if len(self.object_trajectories[idx]) < 5:
                return False
            # Define number of frames to consider for long-term direction change
            num_frames = 8
            object_trajectories_local = copy.deepcopy(self.object_trajectories)
            object_trajectories_local[idx].append([xy[0], xy[1]])
            # Calculate displacements for each frame
            displacements = []
            x_list = object_trajectories_local[idx][1:]
            y_list = object_trajectories_local[idx][:-1]
            for x, y in zip(x_list, y_list):
                displacements.append([a - b for a, b in zip(x, y)])
            # Calculate displacement vectors for each frame
            displacement_vectors = np.array([displacements[i] / np.linalg.norm(displacements[i])
                                             for i in range(len(displacements))])
            # velocity = self.get_id_velocity(idx)
            # Calculate angle between first and last displacement vectors over num_frames
            if len(displacement_vectors) > num_frames:  # and velocity > 7:
                first_displacement_vector = displacement_vectors[0]
                last_displacement_vector = displacement_vectors[-num_frames]
                angle = np.arccos(np.dot(first_displacement_vector, last_displacement_vector) /
                                  (np.linalg.norm(first_displacement_vector) * np.linalg.norm(
                                      last_displacement_vector)))
                change = np.degrees(angle)
                if change > 100 and change:
                    print('Long-term direction change over {} : {:.2f} degrees'.format(idx, change))
                    return True
            else:
                return False

    def detect_vertical_movement(self, idx):
        vertical_pos = [y for x, y in self.object_trajectories[idx][-10:]]

        # Filter out noise using a moving average
        # window_size = 5
        # vertical_pos_filtered = np.convolve(vertical_pos, np.ones(window_size) / window_size, mode='valid')
        # print(type(vertical_pos))
        # print(type(vertical_pos_filtered))
        peaks, _ = find_peaks(np.array(vertical_pos), height=5)
        dips, _ = find_peaks(-np.array(vertical_pos), height=5)

        if peaks.tolist() or dips.tolist():
            # print("idx {}, peaks {}, dips {}".format(idx, peaks, dips))
            return True
        return False

    def update(self, dets, _):
        removed_tracks = set()
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []

        xyxys = dets[0]
        xywh = xyxy2xywh(xyxys)
        confs = dets[1]
        clss = dets[2]

        classes = clss
        position_list, picker_location = self.filter_detections(xywh, clss)

        remain_inds = confs > self.track_thresh
        inds_low = confs > 0.1
        inds_high = confs < self.track_thresh

        position_list_first = []
        for i, cond in enumerate(remain_inds):
            if cond:
                position_list_first.append(position_list[i])

        inds_second = np.logical_and(inds_low, inds_high)
        position_list_second = []
        for i, cond in enumerate(inds_second):
            if cond:
                position_list_second.append(position_list[i])

        dets_second = xywh[inds_second]
        dets = xywh[remain_inds]

        scores_keep = confs[remain_inds]
        scores_second = confs[inds_second]

        clss_keep = classes[remain_inds]
        clss_second = classes[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(xyxy, s, c, pos) for
                          (xyxy, s, c, pos) in zip(dets, scores_keep, clss_keep, position_list_first)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if det.obj_pos != "FOV":
                removed_tracks.add(track.track_id)
                track.mark_removed()
                continue
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(xywh, s, c, pos) for (xywh, s, c, pos) in
                                 zip(dets_second, scores_second, clss_second, position_list_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if det.obj_pos != "FOV":
                removed_tracks.add(track.track_id)
                track.mark_removed()
                track = None
                continue
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            removed_tracks.add(track.track_id)
            track.mark_removed()

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            if track.obj_pos != "FOV":
                removed_tracks.add(track.track_id)
                track.mark_removed()
                track = None
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        
        indices_to_delete = []

        for index, track in enumerate(self.lost_stracks):
            if self.frame_id - track.end_frame > self.max_time_lost:
                removed_tracks.add(track.track_id)
                indices_to_delete.append(index)

        # Now, delete the elements in reverse order to avoid shifting indices
        for index in reversed(indices_to_delete):
            del self.lost_stracks[index]

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                removed_tracks.add(track.track_id)
                track.mark_removed()

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        outputs = []
        for t in output_stracks:
            output = []
            tlwh = t.xywh
            tid = t.track_id
            tlwh = np.expand_dims(tlwh, axis=0)
            xyxy = xywh2xyxy(tlwh)
            xyxy = np.squeeze(xyxy, axis=0)

            output.extend(xyxy)
            output.append(tid)
            output.append(t.cls)
            output.append(t.score)
            outputs.append(output)
        return outputs,removed_tracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb