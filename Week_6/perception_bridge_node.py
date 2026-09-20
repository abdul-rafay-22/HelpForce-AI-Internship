#!/usr/bin/env python3
"""
perception_bridge_node.py — Week 6: HSV vegetation mask -> lateral offset,
with temporal smoothing/hold and a lost-row flag.

Subscribes : /robot/camera/image_raw     (sensor_msgs/Image)
Publishes  : /perception/row_offset      (Float32) pixels; positive = target is
                                          to the RIGHT of image center
             /perception/row_detected    (Bool) False when nothing usable

follow_mode:
  'balance' (default)  Corridor-centering. Compares green mask area on the
                       left half vs right half of the image. More green on
                       the right = robot is closer to the right row = target
                       is to the LEFT (negative offset). Robust for a low,
                       forward-looking camera where rows converge toward the
                       vanishing point (peak-based modes drifted there).
                         offset_px = -balance * balance_scale_px
                         balance   = (right - left) / (right + left) - balance_bias
  'lane'               midpoint of two adjacent column-histogram peaks,
                       tracked frame to frame
  'row'                a column-histogram peak itself, tracked frame to frame

balance_bias (balance mode only): the RAW balance value you observe when the
  robot is physically centered in the aisle (true center = x=0 on the 28-row
  field, since row spacing is symmetric about 0). It is subtracted from the
  raw balance so that a centered robot reads offset ~= 0 instead of a nonzero
  value caused by camera mount/yaw asymmetry. Calibrate it once: park the
  robot at x=0, read the 'bal=' value the node logs (or the published offset),
  and pass that number back as balance_bias. Default 0.0 = no correction.

All modes: EMA smoothing (ema_alpha), hold the last value for up to
max_hold_frames bad frames, then publish detected=False.

Debug: debug_frames/latest_debug.png every N frames
(red = image center, yellow = peaks, green = target).
Float params need a decimal point on the command line (e.g. 300.0).
"""

import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge


def find_peaks_simple(values, min_height, min_distance):
    """Local maxima above min_height, at least min_distance apart (keeps the
    taller of two close peaks). Returns ascending column indices."""
    peaks = []
    for i in range(1, len(values) - 1):
        if values[i] < min_height:
            continue
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
            elif values[i] > values[peaks[-1]]:
                peaks[-1] = i
    return peaks


class PerceptionBridgeNode(Node):
    def __init__(self):
        super().__init__('perception_bridge_node')

        defaults = {
            'hue_low': 42, 'hue_high': 85, 'sat_min': 40, 'val_min': 40,
            'min_peak_height_frac': 0.15,   # fraction of ROI height (peak modes)
            'min_peak_distance_px': 40,
            'smoothing_kernel': 9,
            'roi_top_frac': 0.0,            # 0.0 = full image; 0.4 = bottom 60%
            'follow_mode': 'balance',       # 'balance', 'lane' or 'row'
            'balance_scale_px': 300.0,      # offset (px) at balance = +/-1
            'balance_bias': 0.0,            # raw balance read at true center; subtracted out
            'min_mask_frac': 0.01,          # min green fraction of ROI to trust
            'track_max_jump_px': 120.0,     # peak modes only
            'ema_alpha': 0.5,               # 1.0 = no smoothing
            'max_hold_frames': 5,
            'image_topic': '/robot/camera/image_raw',
            'publish_debug_image': True,
            'debug_image_dir': 'debug_frames',
            'debug_save_every_n_frames': 5,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda n: self.get_parameter(n).value

        self.hue_low, self.hue_high = p('hue_low'), p('hue_high')
        self.sat_min, self.val_min = p('sat_min'), p('val_min')
        self.min_peak_height_frac = p('min_peak_height_frac')
        self.min_peak_distance_px = p('min_peak_distance_px')
        self.smoothing_kernel = p('smoothing_kernel')
        self.roi_top_frac = p('roi_top_frac')
        self.follow_mode = p('follow_mode')
        self.balance_scale_px = float(p('balance_scale_px'))
        self.balance_bias = float(p('balance_bias'))
        self.min_mask_frac = float(p('min_mask_frac'))
        self.track_max_jump_px = p('track_max_jump_px')
        self.ema_alpha = p('ema_alpha')
        self.max_hold_frames = int(p('max_hold_frames'))
        self.publish_debug = p('publish_debug_image')
        self.debug_dir = p('debug_image_dir')
        self.debug_every_n = max(1, int(p('debug_save_every_n_frames')))

        if self.follow_mode not in ('balance', 'lane', 'row'):
            self.get_logger().warn(f"follow_mode '{self.follow_mode}' invalid, using 'balance'")
            self.follow_mode = 'balance'

        self.track_col = None      # smoothed target column (float) or None
        self.miss_count = 0
        self._frame_count = 0

        if self.publish_debug:
            os.makedirs(self.debug_dir, exist_ok=True)
            self.debug_path = os.path.join(self.debug_dir, 'latest_debug.png')

        self.bridge = CvBridge()
        image_topic = p('image_topic')
        self.sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.offset_pub = self.create_publisher(Float32, '/perception/row_offset', 10)
        self.detected_pub = self.create_publisher(Bool, '/perception/row_detected', 10)

        self.get_logger().info(
            f'PerceptionBridgeNode up on {image_topic}, follow_mode={self.follow_mode}, '
            f'hue={self.hue_low}-{self.hue_high}, balance_bias={self.balance_bias:+.3f}')

    # ---- peak modes -------------------------------------------------------
    def _candidates(self, peaks):
        if self.follow_mode == 'row':
            return [float(c) for c in peaks]
        return [0.5 * (a + b) for a, b in zip(peaks[:-1], peaks[1:])]

    def _pick(self, cands, center_col):
        if not cands:
            return None
        if self.track_col is None:
            return min(cands, key=lambda c: abs(c - center_col))
        best = min(cands, key=lambda c: abs(c - self.track_col))
        if abs(best - self.track_col) > self.track_max_jump_px:
            return None
        return best

    # ---- balance mode -----------------------------------------------------
    def _balance_target(self, mask, center_col):
        h, w = mask.shape[:2]
        mid = w // 2
        left = float(np.count_nonzero(mask[:, :mid]))
        right = float(np.count_nonzero(mask[:, mid:]))
        total = left + right
        frac = total / float(h * w)
        if frac < self.min_mask_frac:
            return None, f'mass={frac:.3f}'
        balance = (right - left) / total - self.balance_bias
        target = center_col - balance * self.balance_scale_px
        return target, f'mass={frac:.3f} bal={balance:+.2f}'

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]
        center_col = w / 2.0

        roi = frame[int(self.roi_top_frac * h):, :]
        roi_h = roi.shape[0]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([self.hue_low, self.sat_min, self.val_min])
        upper = np.array([self.hue_high, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        peaks = []
        if self.follow_mode == 'balance':
            measured, info = self._balance_target(mask, center_col)
        else:
            col_sums = np.sum(mask > 0, axis=0).astype(np.float32)
            k = max(1, int(self.smoothing_kernel))
            smoothed = np.convolve(col_sums, np.ones(k, dtype=np.float32) / k, mode='same')
            peaks = find_peaks_simple(
                smoothed, self.min_peak_height_frac * roi_h, self.min_peak_distance_px)
            measured = self._pick(self._candidates(peaks), center_col)
            info = f'peaks={len(peaks)}'

        if measured is not None:
            if self.track_col is None:
                self.track_col = measured
            else:
                a = self.ema_alpha
                self.track_col = a * measured + (1.0 - a) * self.track_col
            self.miss_count = 0
            detected = True
        else:
            self.miss_count += 1
            if self.track_col is not None and self.miss_count <= self.max_hold_frames:
                detected = True            # brief hold of the last value
            else:
                self.track_col = None      # dropped, re-acquire next frame
                detected = False

        offset = float(self.track_col - center_col) if self.track_col is not None else 0.0

        offset_msg, detected_msg = Float32(), Bool()
        offset_msg.data = offset
        detected_msg.data = detected
        self.offset_pub.publish(offset_msg)
        self.detected_pub.publish(detected_msg)

        self.get_logger().info(
            f'mode={self.follow_mode} {info} detected={detected} offset={offset:.1f}px',
            throttle_duration_sec=1.0)

        self._frame_count += 1
        if self.publish_debug and self._frame_count % self.debug_every_n == 0:
            self._save_debug(mask, center_col, peaks)

    def _save_debug(self, mask, center_col, peaks):
        debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        dh = debug.shape[0]
        for c in peaks:
            cv2.line(debug, (int(c), 0), (int(c), dh), (0, 255, 255), 1)
        cv2.line(debug, (int(center_col), 0), (int(center_col), dh), (0, 0, 255), 1)
        if self.track_col is not None:
            cv2.line(debug, (int(self.track_col), 0), (int(self.track_col), dh), (0, 255, 0), 2)
        cv2.imwrite(self.debug_path, debug)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
