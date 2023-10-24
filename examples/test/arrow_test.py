#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

def publish_arrows():
    # Initialize the ROS node
    rospy.init_node('arrows_publisher', anonymous=True)
    
    # Create a MarkerArray
    arrow_markers = MarkerArray()

    # Define a list of start and end points for arrows
    arrow_points = [
        (Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0)),
        (Point(0.0, 0.0, 1.0), Point(0.0, 1.0, 0.0)),
        (Point(1.0, 0.0, 2.0), Point(0.0, 0.0, 1.0))
    ]

    for idx, (start_point, end_point) in enumerate(arrow_points):
        # Create an Arrow Marker
        arrow = Marker()
        arrow.header.frame_id = "base_link"  # Set the frame of reference
        arrow.type = Marker.ARROW  # Arrow marker type
        arrow.action = Marker.ADD
        arrow.scale.x = 0.1  # Arrow width
        arrow.scale.y = 0.2  # Arrow head width
        arrow.scale.z = 0.6  # Arrow head length
        arrow.color.r = 1.0  # Red color
        arrow.color.a = 1.0  # Fully opaque
        arrow.pose.orientation.w = 1.0

        # Define the starting point
        arrow.points.append(start_point)

        # Define the ending point
        end_point = Point()
        end_point.x  = start_point.x + 1
        end_point.y  = start_point.y + 1
        end_point.z  = start_point.z + 1
        arrow.points.append(end_point)

        # Add the arrow to the MarkerArray
        arrow.id = idx
        arrow_markers.markers.append(arrow)

    # Create a publisher for the MarkerArray
    marker_pub = rospy.Publisher('arrow_marker_array', MarkerArray, queue_size=10)

    # Publish the MarkerArray
    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        marker_pub.publish(arrow_markers)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_arrows()
    except rospy.ROSInterruptException:
        pass
