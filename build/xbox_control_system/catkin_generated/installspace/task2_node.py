#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import cv2
from ultralytics import YOLO
from playsound import playsound
import threading

class Task2ObjectDetection:
    def __init__(self):
        rospy.init_node('task2_node')
        rospy.Subscriber('/xbox_button', String, self.button_callback)
        self.model = YOLO("/home/robot/ball/train/weights/best.pt")
        self.audio_map = {
            "blue": "/home/robot/ball/blue.mp3",
            "redball": "/home/robot/ball/redball.mp3"
        }
        self.task_active = False
        self.thread = None
        self.stop_event = threading.Event()
        rospy.loginfo("Task2 node ready. Waiting for 'B' button to start detection.")

    def button_callback(self, msg):
        if msg.data == 'B':
            if not self.task_active:
                rospy.loginfo("Button B received. Starting object detection.")
                self.stop_event.clear()
                self.thread = threading.Thread(target=self.run_detection)
                self.thread.start()
                self.task_active = True
            else:
                rospy.loginfo("Task2 already running.")
        else:
            if self.task_active:
                rospy.loginfo(f"Received button '{msg.data}' - stopping Task2.")
                self.stop_event.set()
                if self.thread:
                    self.thread.join()
                self.task_active = False

    def play_audio(self, label):
        if label in self.audio_map:
            threading.Thread(target=playsound, args=(self.audio_map[label],), daemon=True).start()

    def run_detection(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            rospy.logerr("Error: Could not open camera.")
            self.task_active = False
            return

        rospy.loginfo("Camera opened. Running YOLOv8 detection.")

        while not rospy.is_shutdown() and not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                rospy.logwarn("Failed to grab frame")
                break

            results = self.model(frame, conf=0.3)[0]

            if results.boxes:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = self.model.names[cls]

                    self.play_audio(label)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Live Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.loginfo("Exiting detection loop (q pressed).")
                break

        cap.release()
        cv2.destroyAllWindows()
        self.task_active = False
        rospy.loginfo("Detection task ended.")

if __name__ == '__main__':
    try:
        Task2ObjectDetection()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

