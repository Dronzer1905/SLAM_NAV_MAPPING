#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import serial
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import threading

class Task3PersonFollower:
    def __init__(self):
        rospy.init_node('task3_node')
        rospy.Subscriber('/xbox_button', String, self.button_callback)

        self.arduino = None
        self.model = YOLO("yolov8n.pt")

        self.pipeline = None
        self.align = None
        self.intrinsics = None

        self.task_active = False
        self.thread = None
        self.stop_event = threading.Event()

        rospy.loginfo("Task3 node ready. Waiting for 'X' button to start person following.")

    def button_callback(self, msg):
        if msg.data == 'X':
            if not self.task_active:
                rospy.loginfo("Button X received. Starting person follower.")
                self.stop_event.clear()
                self.thread = threading.Thread(target=self.run_follower)
                self.thread.start()
                self.task_active = True
            else:
                rospy.loginfo("Task3 already running.")
        else:
            if self.task_active:
                rospy.loginfo(f"Received button '{msg.data}' - stopping Task3.")
                self.stop_event.set()
                if self.thread:
                    self.thread.join()
                self.task_active = False

    def send_command(self, cmd):
        rospy.loginfo(f"[Arduino] Command: {cmd}")
        if self.arduino and self.arduino.is_open:
            self.arduino.write(cmd.encode())

    def get_closest_person(self, results, depth_frame):
        closest_dist = float('inf')
        best_box = None
        best_xyz = None

        for result in results.boxes:
            if int(result.cls[0]) == 0:  # person class = 0
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                dist = depth_frame.get_distance(cx, cy)

                if dist > 0 and dist < closest_dist:
                    closest_dist = dist
                    xyz = rs.rs2_deproject_pixel_to_point(self.intrinsics, [cx, cy], dist)
                    best_box = (x1, y1, x2, y2)
                    best_xyz = xyz

        return best_box, best_xyz

    def run_follower(self):
        try:
            self.arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        except Exception as e:
            rospy.logerr(f"Failed to open serial port: {e}")
            self.task_active = False
            return

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        try:
            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            rospy.loginfo("RealSense pipeline started.")
        except Exception as e:
            rospy.logerr(f"Failed to start RealSense pipeline: {e}")
            self.arduino.close()
            self.task_active = False
            return

        while not rospy.is_shutdown() and not self.stop_event.is_set():
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            results = self.model(color_image)[0]

            box, xyz = self.get_closest_person(results, depth_frame)

            if box is not None and xyz is not None:
                x1, y1, x2, y2 = box
                X, Y, Z = xyz
                label = f"X={X:.2f} Z={Z:.2f}"
                cv2.putText(color_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if Z > 0.6:
                    if abs(X) > 0.2:
                        self.send_command("L" if X < 0 else "R")
                    else:
                        self.send_command("F")
                elif Z < 0.4:
                    self.send_command("B")
                else:
                    self.send_command("S")
            else:
                self.send_command("S")

            cv2.imshow("Person Follower", color_image)
            cv2.imshow("Depth", depth_colormap)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.loginfo("Exiting person follower loop (q pressed).")
                break

        self.pipeline.stop()
        if self.arduino.is_open:
            self.arduino.close()
        cv2.destroyAllWindows()
        self.task_active = False
        rospy.loginfo("Person follower task ended.")

if __name__ == '__main__':
    try:
        Task3PersonFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

