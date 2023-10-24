import tf2_ros
import tf
import numpy as np
import rospy

def transform_to_matrix(transform):
    translation = [transform.transform.translation.x,
                   transform.transform.translation.y,
                   transform.transform.translation.z]

    rotation = [transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w]

    rotation_matrix = tf.transformations.quaternion_matrix(rotation)

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation

    transform_matrix = np.dot(translation_matrix, rotation_matrix)

    return transform_matrix

def get_world_T_cam():
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    transform = tf_buffer.lookup_transform("world", "rgb_camera_link", rospy.Time(0), rospy.Duration(1.0))
    return transform_to_matrix(transform)