#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import evdev
import serial
import time
import threading

DEVICE_PATH = '/dev/input/event2'
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
DEADZONE = 3000

class Task1Controller:
    def __init__(self):
        rospy.init_node('task1_node')
        rospy.Subscriber('/xbox_button', String, self.button_callback)
        self.task_active = False
        self.thread = None
        self.stop_event = threading.Event()
        rospy.loginfo("Task1 node ready and waiting for A button to start.")

    def button_callback(self, msg):
        if msg.data == 'A':
            if not self.task_active:
                rospy.loginfo("A button received. Starting joystick control.")
                self.stop_event.clear()
                self.thread = threading.Thread(target=self.joystick_control_loop)
                self.thread.start()
                self.task_active = True
            else:
                rospy.loginfo("Task1 already running.")
        else:
            # Stop task if running and different button received
            if self.task_active:
                rospy.loginfo(f"Received button '{msg.data}' - stopping Task1.")
                self.stop_event.set()
                if self.thread:
                    self.thread.join()
                self.task_active = False

    def joystick_control_loop(self):
        try:
            device = evdev.InputDevice(DEVICE_PATH)
            rospy.loginfo(f"Listening to joystick: {device.name}")
        except Exception as e:
            rospy.logerr(f"Failed to access device at {DEVICE_PATH}: {e}")
            self.task_active = False
            return

        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            time.sleep(2)
            rospy.loginfo("Serial connected.")
        except Exception as e:
            rospy.logerr(f"Failed to open serial port {SERIAL_PORT}: {e}")
            self.task_active = False
            return

        AXIS_X = evdev.ecodes.ABS_X
        AXIS_Y = evdev.ecodes.ABS_Y

        x = 32768
        y = 32768
        last_cmd = ''

        for event in device.read_loop():
            if self.stop_event.is_set() or rospy.is_shutdown():
                rospy.loginfo("Stopping joystick control loop.")
                break

            if event.type == evdev.ecodes.EV_ABS:
                if event.code == AXIS_X:
                    x = event.value
                elif event.code == AXIS_Y:
                    y = event.value

                x_centered = x - 32768
                y_centered = y - 32768

                # Deadzone
                if abs(x_centered) < DEADZONE:
                    x_centered = 0
                if abs(y_centered) < DEADZONE:
                    y_centered = 0

                # Determine command
                if x_centered == 0 and y_centered == 0:
                    cmd = 'S'
                elif abs(x_centered) > abs(y_centered):
                    cmd = 'R' if x_centered > 0 else 'L'
                else:
                    cmd = 'F' if y_centered < 0 else 'B'

                if cmd != last_cmd:
                    rospy.loginfo(f"Sending: {cmd}")
                    ser.write(cmd.encode())
                    last_cmd = cmd

        ser.close()
        self.task_active = False
        rospy.loginfo("Joystick control thread ended.")

if __name__ == '__main__':
    try:
        Task1Controller()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

