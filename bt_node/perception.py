import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2


class SimplePersonDetector(Node):
    def __init__(self):
        super().__init__('simple_person_detector')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image,
            '/head_front_camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.person_pub = self.create_publisher(Bool, '/person_detected', 10)

        # Initialize HOG descriptor with default people detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.get_logger().info("SimplePersonDetector node started.")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        # Detect people in the image
        (rects, _) = self.hog.detectMultiScale(
            frame,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05
        )

        person_found = len(rects) > 0
        self.person_pub.publish(Bool(data=person_found))

        if person_found:
            self.get_logger().info("Person detected.")
        else:
            self.get_logger().info("No person detected.")

def main(args=None):
    rclpy.init(args=args)
    node = SimplePersonDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()