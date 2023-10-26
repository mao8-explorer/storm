#!/usr/bin/env python

import rospy
from interactive_markers.interactive_marker_server import *

from visualization_msgs.msg import Marker , InteractiveMarkerControl

def control_marker_pose(x, y, z):
    server = InteractiveMarkerServer("marker_control")

    int_marker = InteractiveMarker()
    int_marker.header.frame_id = "base_link"
    int_marker.name = "my_marker"
    int_marker.description = "Interactive Marker"

    box_marker = Marker()
    box_marker.type = Marker.CUBE
    box_marker.scale.x = 0.2
    box_marker.scale.y = 0.2
    box_marker.scale.z = 0.2
    box_marker.color.r = 0.0
    box_marker.color.g = 1.0
    box_marker.color.b = 0.0
    box_marker.color.a = 1.0

    control = InteractiveMarkerControl()
    control.always_visible = True
    control.markers.append(box_marker)
    int_marker.controls.append(control)

    server.insert(int_marker, lambda feedback: None)
    server.applyChanges()

    # Update the position of the marker
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        int_marker.pose.position.x = x
        int_marker.pose.position.y = y
        int_marker.pose.position.z = z
        server.insert(int_marker, lambda feedback: None)
        server.applyChanges()
        rate.sleep()
        x += 0.1
        y += 0.1
        z += 0.1

if __name__ == "__main__":
    rospy.init_node("interactive_marker_control_node")

    x = 0.0
    y = 0.0
    z = 0.0

    try:
        x = float(input("Enter X position: "))
        y = float(input("Enter Y position: "))
        z = float(input("Enter Z position: "))
    except ValueError:
        rospy.logerr("Invalid input. Using default position (0, 0, 0).")

    control_marker_pose(x, y, z)
