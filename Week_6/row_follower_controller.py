#!/usr/bin/env python3
"""
row_follower_controller.py — Week 6

PD row-following controller (straight rows) with a LATCHED END-OF-ROW stop.

What this adds over the Week 5 version
--------------------------------------
The Week 5 controller already stopped when perception reported no row for
`max_missed_ticks` consecutive ticks — but that was a *momentary* safety
stop: if green flickered back, it resumed. That conflates two different
situations:

  * brief loss mid-aisle (a gap between plants, a dead plant, a glare
    frame) — the robot SHOULD hold heading and keep going; and
  * the actual END OF THE ROW — green falls away for good because the crop
    rows have ended — the robot SHOULD stop, deliberately, and stay stopped.

This version distinguishes them with a longer, separate counter and a
LATCH. Two thresholds now:

  * `max_missed_ticks` (unchanged, short ~0.5 s): brief loss -> zero
    velocity but NOT latched; if the row comes back the robot resumes.
  * `end_of_row_ticks` (new, longer ~2 s of sustained no-detection):
    treated as the row having ended -> latch into an END_OF_ROW state,
    publish zero velocity, and STAY there. Only a process restart clears
    the latch. This is the deliberate, reportable "row complete" stop, and
    it's the building block the later headland-turn behavior will trigger
    from (turn instead of latch).

The PID, the offset smoothing, the safety stop, and all topics are
otherwise unchanged from the version that logged the straight-row baseline,
so the tuned behavior you validated is preserved.

Topics:
  Subscribes: /perception/row_offset, /perception/row_detected
  Publishes:  /cmd_vel (geometry_msgs/Twist)

Run:
  python3 row_follower_controller.py --ros-args -p kp:=0.004 -p kd:=0.001
Tune the end-of-row sensitivity if needed (longer = more tolerant of long
in-row gaps before it decides the row has ended):
  python3 row_follower_controller.py --ros-args -p end_of_row_ticks:=50
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist


class RowFollowerController(Node):
    def __init__(self):
        super().__init__('row_follower_controller')

        self.declare_parameter('kp', 0.02)
        self.declare_parameter('kd', 0.01)
        self.declare_parameter('linear_speed', 0.4)
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('max_missed_ticks', 10)      # ~0.5s at 20Hz: brief loss
        self.declare_parameter('end_of_row_ticks', 40)      # ~2.0s at 20Hz: row has ended
        self.declare_parameter('offset_filter_alpha', 0.3)  # 0=heavy smoothing, 1=none
        self.declare_parameter('offset_topic', '/perception/row_offset')
        self.declare_parameter('detected_topic', '/perception/row_detected')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        self.kp = self.get_parameter('kp').value
        self.kd = self.get_parameter('kd').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.max_missed_ticks = self.get_parameter('max_missed_ticks').value
        self.end_of_row_ticks = self.get_parameter('end_of_row_ticks').value
        self.filter_alpha = self.get_parameter('offset_filter_alpha').value
        rate_hz = self.get_parameter('control_rate_hz').value

        self.prev_error = 0.0
        self.prev_time = self.get_clock().now()
        self.filtered_offset = 0.0
        self.last_detected = False
        self.missed_ticks = 0
        self.end_of_row_latched = False   # once True, stays stopped

        self.offset_sub = self.create_subscription(
            Float32, self.get_parameter('offset_topic').value, self.offset_cb, 10)
        self.detected_sub = self.create_subscription(
            Bool, self.get_parameter('detected_topic').value, self.detected_cb, 10)
        self.cmd_pub = self.create_publisher(
            Twist, self.get_parameter('cmd_vel_topic').value, 10)
        self.timer = self.create_timer(1.0 / rate_hz, self.control_loop)

        self.get_logger().info(
            f'RowFollowerController up. kp={self.kp} kd={self.kd} '
            f'linear_speed={self.linear_speed} rate={rate_hz}Hz '
            f'offset_filter_alpha={self.filter_alpha} '
            f'max_missed_ticks={self.max_missed_ticks} '
            f'end_of_row_ticks={self.end_of_row_ticks}'
        )

    def offset_cb(self, msg: Float32):
        a = self.filter_alpha
        self.filtered_offset = a * msg.data + (1.0 - a) * self.filtered_offset

    def detected_cb(self, msg: Bool):
        self.last_detected = msg.data
        if msg.data:
            self.missed_ticks = 0

    def control_loop(self):
        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds / 1e9
        self.prev_time = now
        if dt <= 0.0:
            dt = 1e-3

        cmd = Twist()  # defaults to all-zero = stop

        # --- Latched end-of-row: once we've decided the row ended, stay stopped.
        if self.end_of_row_latched:
            self.cmd_pub.publish(cmd)
            return

        # Count consecutive ticks with no valid detection.
        if not self.last_detected:
            self.missed_ticks += 1

        # --- Sustained loss => the row has ended. Latch and stop for good.
        if self.missed_ticks >= self.end_of_row_ticks:
            self.end_of_row_latched = True
            self.get_logger().info(
                f'END OF ROW: no detection for {self.missed_ticks} ticks '
                f'(>= {self.end_of_row_ticks}). Stopping and latching.'
            )
            self.cmd_pub.publish(cmd)  # zero velocity
            return

        # --- Brief loss (shorter than end-of-row) => momentary safety stop,
        #     NOT latched. If the row returns, we resume next tick.
        if self.missed_ticks > self.max_missed_ticks:
            self.get_logger().warn(
                f'Row briefly lost ({self.missed_ticks} ticks) — holding stop.',
                throttle_duration_sec=1.0,
            )
            self.cmd_pub.publish(cmd)
            return

        # --- Normal PD row following.
        error = self.filtered_offset
        derivative = (error - self.prev_error) / dt
        angular_z = -(self.kp * error + self.kd * derivative)
        self.prev_error = error

        cmd.linear.x = self.linear_speed
        cmd.angular.z = angular_z
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = RowFollowerController()
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
