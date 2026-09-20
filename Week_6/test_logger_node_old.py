#!/usr/bin/env python3
"""
test_logger_node.py — Week 5, Checklist Phase 4 + Phase 5

Logs ONE closed-loop test run: robot position vs. the row centerline over
time, and the pass/fail result, appended to a results CSV. Start it right
after you place the robot at a row entrance and start the controller —
it exits on its own when the run finishes (success, left the row, or
timed out), so one run = one command.

Requires an /odom publisher (nav_msgs/Odometry) from the robot. If you
don't have one yet, add "Isaac Compute Odometry" -> "ROS2 Publish
Odometry" nodes to the robot's Action Graph (same graph your camera
publishing already lives in) before running this — see the guide, Phase 3.

BEFORE YOU TRUST ANY LOGGED RESULT: this node assumes /odom's y-coordinate
is the axis that increases as the robot drives down a row (that's what
row_end_y checks against). That assumption has never actually been
verified against your Isaac ROS2 bridge. Drive the robot forward a few
seconds and `ros2 topic echo /odom` first — if x is the moving axis
instead of y, every run this logs will silently check the wrong axis and
the success/fail column will be meaningless.

Usage example — UPDATED for your current field (generate_week6_field.py:
28 rows @ 1.8m spacing, row length 14m, robot starting at the row
entrance). Row spacing is symmetric around x=0, so the middle aisle
(between the two center rows) sits at x=0.0 — this replaces the old
row 2 / -0.9m example from the 6-row Week 5 field, which no longer
matches your field's geometry:

  python3 test_logger_node.py --ros-args \
    -p row_centerline_x:=0.0 \
    -p row_end_y:=7.0 \
    -p fail_deviation_m:=0.6 \
    -p timeout_sec:=60.0 \
    -p run_label:=baseline_middle_aisle \
    -p results_csv:=week6_results_baseline.csv

To test a different row instead of the middle aisle, that row's centerline
x is: start_x + row_index * 1.8, where start_x = -((28-1) * 1.8) / 2 =
-24.3, and row_index counts from 0.
"""

import csv
import os
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class TestLoggerNode(Node):
    def __init__(self):
        super().__init__('test_logger_node')

        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('row_centerline_x', 0.0)
        self.declare_parameter('row_end_y', 7.0)
        self.declare_parameter('fail_deviation_m', 0.6)
        self.declare_parameter('timeout_sec', 60.0)
        self.declare_parameter('run_label', 'run')
        self.declare_parameter('results_csv', 'week5_results.csv')
        self.declare_parameter('per_tick_log_dir', 'week5_run_logs')

        self.centerline_x = self.get_parameter('row_centerline_x').value
        self.row_end_y = self.get_parameter('row_end_y').value
        self.fail_deviation = self.get_parameter('fail_deviation_m').value
        self.timeout_sec = self.get_parameter('timeout_sec').value
        self.run_label = self.get_parameter('run_label').value
        self.results_csv = self.get_parameter('results_csv').value
        self.per_tick_dir = self.get_parameter('per_tick_log_dir').value

        os.makedirs(self.per_tick_dir, exist_ok=True)
        self.per_tick_path = os.path.join(self.per_tick_dir, f'{self.run_label}.csv')
        self.per_tick_file = open(self.per_tick_path, 'w', newline='')
        self.per_tick_writer = csv.writer(self.per_tick_file)
        self.per_tick_writer.writerow(['t_sec', 'x', 'y', 'deviation_m'])

        self.start_time = time.time()
        self.deviations = []
        self.finished = False

        self.sub = self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self.odom_cb, 20)
        self.timeout_timer = self.create_timer(0.5, self.check_timeout)

        self.get_logger().info(
            f'TestLoggerNode up. run_label={self.run_label} '
            f'centerline_x={self.centerline_x} row_end_y={self.row_end_y} '
            f'fail_deviation_m={self.fail_deviation} timeout_sec={self.timeout_sec}'
        )

    def odom_cb(self, msg: Odometry):
        if self.finished:
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        deviation = x - self.centerline_x
        t = time.time() - self.start_time

        self.per_tick_writer.writerow([f'{t:.3f}', f'{x:.4f}', f'{y:.4f}', f'{deviation:.4f}'])
        self.deviations.append(deviation)

        if abs(deviation) > self.fail_deviation:
            self.finish_run('FAIL_LEFT_ROW', t)
        elif y >= self.row_end_y:
            self.finish_run('SUCCESS', t)

    def check_timeout(self):
        if self.finished:
            return
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_sec:
            self.finish_run('FAIL_TIMEOUT', elapsed)

    def finish_run(self, result, duration):
        self.finished = True
        self.per_tick_file.close()

        max_dev = max((abs(d) for d in self.deviations), default=0.0)
        mean_dev = (sum(abs(d) for d in self.deviations) / len(self.deviations)) if self.deviations else 0.0

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


if __name__ == '__main__':
    main()
