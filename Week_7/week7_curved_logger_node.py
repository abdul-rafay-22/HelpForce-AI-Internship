#!/usr/bin/env python3
"""
week7_curved_logger_node.py

Deviation logger for the Week 7 curved+uneven field. This is NOT a drop-in
replacement for test_logger_node.py -- keep that one for re-running the
straight-row baseline. Use THIS one only on week7_field_curved_uneven.usd.

WHY A SEPARATE LOGGER: on a curved field the correct path is not x=0, it's
a moving centerline curve_offset(z) that shifts by up to 0.9 m across the
run. A logger that measures deviation from a fixed x=0 (like Week 6's) would
mark a PERFECTLY driven curved run as up to 0.9 m off-track -- worse than
the 0.6 m fail threshold -- and fail every run regardless of how good the
controller is. This node measures deviation from the actual curve instead.

CURVE_AMPLITUDE / CURVE_WAVELENGTH / CURVE_PHASE_Z below MUST exactly match
the same constants in generate_week7_field_curved_uneven.py. If you tune the
field's curve, tune these to match -- they are intentionally duplicated
(not imported) so this file has no dependency on where the field generator
script lives.

Usage (matches the Week 6 logger's parameter convention):
    ros2 run <your_package> week7_curved_logger_node.py --ros-args \\
        -p run_length_m:=10.0 \\
        -p start_offset_m:=0.0 \\
        -p fail_deviation_m:=0.6 \\
        -p timeout_sec:=90.0 \\
        -p results_csv:=week7_results_baseline_curved_uneven.csv

Or, if not wired into a package yet, run directly with:
    python3 week7_curved_logger_node.py --ros-args -p run_length_m:=10.0 ...

Result columns match Week 6's convention plus one addition (curve_dev_m is
the actual signed number this node uses for pass/fail; recorded alongside
raw_x/curve_x so you can sanity-check the curve math independently):
    timestamp,result,max_dev,mean_dev,elapsed_sec,final_z,raw_x,curve_x,note
"""

import csv
import math
import os
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

# --- MUST match generate_week7_field_curved_uneven.py exactly ---
CURVE_AMPLITUDE = 0.9
CURVE_WAVELENGTH = 14.0
CURVE_PHASE_Z = 6.0


def curve_offset(z: float) -> float:
    return CURVE_AMPLITUDE * math.sin(2 * math.pi * (z - CURVE_PHASE_Z) / CURVE_WAVELENGTH)


class Week7CurvedLogger(Node):
    def __init__(self):
        super().__init__('week7_curved_logger_node')

        self.declare_parameter('run_length_m', 10.0)
        self.declare_parameter('start_offset_m', 0.0)
        self.declare_parameter('fail_deviation_m', 0.6)
        self.declare_parameter('timeout_sec', 90.0)
        self.declare_parameter('results_csv', 'week7_results_baseline_curved_uneven.csv')
        self.declare_parameter('start_z', 6.0)  # must match the robot's actual start pose z

        self.run_length_m = self.get_parameter('run_length_m').value
        self.start_offset_m = self.get_parameter('start_offset_m').value
        self.fail_deviation_m = self.get_parameter('fail_deviation_m').value
        self.timeout_sec = self.get_parameter('timeout_sec').value
        self.results_csv = self.get_parameter('results_csv').value
        self.start_z = self.get_parameter('start_z').value

        self.target_final_z = self.start_z - self.run_length_m

        self.devs = []
        self.start_time = None
        self.finished = False
        self.last_x = 0.0
        self.last_z = self.start_z

        self.sub = self.create_subscription(Odometry, '/odom', self.on_odom, 20)
        self.timer = self.create_timer(0.5, self.check_timeout)

        self.get_logger().info(
            f"week7_curved_logger_node ready: run_length_m={self.run_length_m}, "
            f"fail_deviation_m={self.fail_deviation_m}, timeout_sec={self.timeout_sec}, "
            f"start_z={self.start_z} -> target_final_z={self.target_final_z}, "
            f"results_csv={self.results_csv}"
        )
        self.get_logger().warn(
            "Confirm CURVE_AMPLITUDE/CURVE_WAVELENGTH/CURVE_PHASE_Z in this file match "
            "the field generator EXACTLY -- if you changed the field's curve params and "
            "forgot to update this file, every run will score against the wrong curve."
        )

    def on_odom(self, msg: Odometry):
        if self.finished:
            return

        if self.start_time is None:
            self.start_time = time.time()

        x = msg.pose.pose.position.x
        z = msg.pose.pose.position.z
        self.last_x = x
        self.last_z = z

        expected_x = curve_offset(z) + self.start_offset_m
        dev = abs(x - expected_x)
        self.devs.append(dev)

        if dev > self.fail_deviation_m:
            self.finish('FAIL', dev, x, expected_x, note='exceeded fail_deviation_m')
            return

        if z <= self.target_final_z:
            self.finish('SUCCESS', dev, x, expected_x, note='reached run_length_m')
            return

    def check_timeout(self):
        if self.finished or self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_sec:
            expected_x = curve_offset(self.last_z) + self.start_offset_m
            self.finish('FAIL_TIMEOUT', abs(self.last_x - expected_x), self.last_x, expected_x,
                        note=f'exceeded timeout_sec ({self.timeout_sec}s)')

    def finish(self, result, final_dev, raw_x, curve_x, note=''):
        self.finished = True
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        max_dev = max(self.devs) if self.devs else final_dev
        mean_dev = sum(self.devs) / len(self.devs) if self.devs else final_dev

        write_header = not os.path.exists(self.results_csv)
        with open(self.results_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['timestamp', 'result', 'max_dev', 'mean_dev', 'elapsed_sec',
                                  'final_z', 'raw_x', 'curve_x', 'note'])
            writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), result, f'{max_dev:.4f}',
                              f'{mean_dev:.4f}', f'{elapsed:.2f}', f'{self.last_z:.3f}',
                              f'{raw_x:.4f}', f'{curve_x:.4f}', note])

        self.get_logger().info(
            f"RESULT: {result}  max_dev={max_dev:.3f}m  mean_dev={mean_dev:.3f}m  "
            f"elapsed={elapsed:.1f}s  final_z={self.last_z:.2f}  ({note})"
        )
        self.get_logger().info(f"Written to {self.results_csv}. Ctrl+C to exit.")


def main():
    rclpy.init()
    node = Week7CurvedLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
