import rospy
import time
from std_msgs.msg import Float32


def demo():
    # Set the threshold value
    threshold = 5
    start = time.time()

    # Start the loop
    print("what happened --- ")
    while not rospy.is_shutdown():
        try:
            # Check if the threshold has been reached
            if time.time() - start > threshold:
                print("Threshold reached")
                break
            else:
                print("Waiting...{}".format(time.time() - start))
                time.sleep(1)
        except KeyboardInterrupt:
            print("Program terminated")
            return False
    print("what happened")
    return True

def demo_callback():
    pass

if __name__ == "__main__":
    rospy.init_node("demo")
    rospy.Subscriber("demo",Float32, demo_callback)
    rospy.Publisher("want",Float32, demo_callback)
    while not rospy.is_shutdown():
        try:
            if not demo() : break
        except KeyboardInterrupt:
            print("Program terminated ---")
            break
    

    print("aLL OVER")