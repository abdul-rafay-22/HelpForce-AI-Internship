#!/usr/bin/env python3
"""
test_logger_node.py — Week 6: logs ONE closed-loop run to a results CSV.

Fixes the Week 5 version's axis assumption (it treated odom y as "forward"
and odom x as "lateral"; on this robot forward is odom x). This version
takes the FIRST /odom message as the run origin and projects all motion onto
the robot's initial heading, so it does not depend on which odom axis the
robot happens to drive along:

  forward_m   = distance travelled along the starting heading
  lateral_m   = sideways displacement from the start line (+ = LEFT)
  deviation_m = start_offset_m + lateral_m   (+ = left of the row centerline)

Place the robot at the row start, facing down the aisle, start the
controller, then start this node. It exits on its own when the run ends.

  SUCCESS        forward_m >= run_length_m
  FAIL_LEFT_ROW  |deviation_m| > fail_deviation_m
  FAIL_TIMEOUT   timeout_sec elapsed
  FAIL_NO_ODOM   no /odom message ever arrived

Example (28-row Week 6 field: rows run along world Z, 14 m long, middle aisle
at x=0; start at Z=-6 facing +Z and run 11 m):

  python3 test_logger_node.py --ros-args \
    -p run_length_m:=11.0 \
    -p fail_deviation_m:=0.6 \
    -p timeout_sec:=60.0 \
    -p run_label:=baseline_middle_aisle_1 \
    -p results_csv:=week6_results_baseline.csv

Set start_offset_m if the robot does not start on the row centerline
(positive = starts LEFT of the centerline). Use decimals for float params.
"""

import csv
import math
import os
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class TestLoggerNode(Node):
    def __init__(self):
        super().__init__('test_logger_node')

        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('run_length_m', 11.0)
        self.declare_parameter('start_offset_m', 0.0)
        self.declare_parameter('fail_deviation_m', 0.6)
        self.declare_parameter('timeout_sec', 60.0)
        self.declare_parameter('run_label', 'run')
        self.declare_parameter('results_csv', 'week6_results.csv')
        self.declare_parameter('per_tick_log_dir', 'week6_run_logs')

        self.run_length = self.get_parameter('run_length_m').value
        self.start_offset = self.get_parameter('start_offset_m').value
        self.fail_deviation = self.get_parameter('fail_deviation_m').value
        self.timeout_sec = self.get_parameter('timeout_sec').value
        self.run_label = self.get_parameter('run_label').value
        self.results_csv = self.get_parameter('results_csv').value
        self.per_tick_dir = self.get_parameter('per_tick_log_dir').value

        os.makedirs(self.per_tick_dir, exist_ok=True)
        self.per_tick_path = os.path.join(self.per_tick_dir, f'{self.run_label}.csv')
        self.per_tick_file = open(self.per_tick_path, 'w', newline='')
        self.per_tick_writer = csv.writer(self.per_tick_file)
        self.per_tick_writer.writerow(['t_sec', 'forward_m', 'lateral_m', 'deviation_m'])

        self.origin = None          # (x0, y0, cos_yaw0, sin_yaw0) from first odom msg
        self.start_time = time.time()
        self.deviations = []
        self.finished = False

        self.sub = self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self.odom_cb, 20)
        self.timeout_timer = self.create_timer(0.5, self.check_timeout)

        self.get_logger().info(
            f'TestLoggerNode up. run_label={self.run_label} '
            f'run_length_m={self.run_length} fail_deviation_m={self.fail_deviation} '
            f'timeout_sec={self.timeout_sec} start_offset_m={self.start_offset}'
        )

    def odom_cb(self, msg: Odometry):
        if self.finished:
            return
        p = msg.pose.pose.position

        if self.origin is None:
            yaw0 = yaw_from_quat(msg.pose.pose.orientation)
            self.origin = (p.x, p.y, math.cos(yaw0), math.sin(yaw0))
            self.start_time = time.time()   # clock starts at the first odom msg
            self.get_logger().info(
                f'Run origin captured: odom=({p.x:.3f}, {p.y:.3f}) yaw={math.degrees(yaw0):.1f} deg')

        x0, y0, c, s = self.origin
        dx, dy = p.x - x0, p.y - y0
        forward = dx * c + dy * s
        lateral = -dx * s + dy * c
        deviation = self.start_offset + lateral
        t = time.time() - self.start_time

        self.per_tick_writer.writerow(
            [f'{t:.3f}', f'{forward:.4f}', f'{lateral:.4f}', f'{deviation:.4f}'])
        self.deviations.append(deviation)

        self.get_logger().info(
            f'fwd={forward:.2f} m  dev={deviation:+.3f} m', throttle_duration_sec=2.0)

        if abs(deviation) > self.fail_deviation:
            self.finish_run('FAIL_LEFT_ROW', t)
        elif forward >= self.run_length:
            self.finish_run('SUCCESS', t)

    def check_timeout(self):
        if self.finished:
            return
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_sec:
            self.finish_run('FAIL_NO_ODOM' if self.origin is None else 'FAIL_TIMEOUT', elapsed)

    def finish_run(self, result, duration):
        self.finished = True
        self.per_tick_file.close()

        max_dev = max((abs(d) for d in self.deviations), default=0.0)
        mean_dev = (sum(abs(d) for d in self.deviations) / len(self.deviations)
                    if self.deviations else 0.0)

        write_header = not os.path.exists(self.results_csv)
        with open(self.results_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['run_label', 'result', 'duration_sec', 'max_deviation_m',
                                 'mean_deviation_m', 'per_tick_log'])
            writer.writerow([self.run_label, result, f'{duration:.2f}', f'{max_dev:.4f}',
                             f'{mean_dev:.4f}', self.per_tick_path])

        self.get_logger().info(
            f'Run "{self.run_label}" finished: {result}  '
            f'max_dev={max_dev:.3f}m mean_dev={mean_dev:.3f}m duration={duration:.1f}s'
        )
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = TestLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not node.finished:
            node.per_tick_file.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
