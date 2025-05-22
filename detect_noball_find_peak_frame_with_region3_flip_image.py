import torch
from ultralytics import YOLO, SAM
import cv2
import numpy as np
import time
import logging
import psutil
import os
from datetime import datetime
# import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from shapely.geometry import Point, Polygon
from trackers import SORTTracker
# Add supervision import for Detections
import supervision as sv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('no_ball_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NoBallDetector:
    def __init__(self, yolo_model_path, shoe_model_path, seg_model_path, input_path, input_type='video', right_to_left=False, fps=30, skip_seconds=3):
        self.line_points = []
        self.right_to_left = right_to_left
        self.max_y_persistent_peak = None
        self.max_y_persistent_centroid = None
        self.persistent_centroid = None
        self.heel_point = None
        self.heel_points = []
        self.toe_point = None  # Added toe point
        self.toe_points = []  # Track toe points over frames
        self.min_centroids = []
        self.bowler_bottom_rights = []
        self.max_peak_y = None
        self.stump_bottom_left = None
        self.polygon_pts = None
        self.prev_centroid = None
        self.persistent_counter = 0
        self.bowler_bottom_right = None
        self.prev_bowler_bottom_right = None
        self.prev_bowler_bottom_left = None
        self.prev_bowler_bottom_center = None
        self.detect_persistency = False
        self.prev_side = None
        self.curr_side = None
        self.bowler_crossed_line = False
        self.bowler_bottom_left = None
        self.bowler_bottom_points = None
        self.bowler_returning = False
        self.prompted_bbox = []
        self.shoe_detected = False
        self.prev_shoe_box = None
        self.prev_fielder_box = None
        self.iou_threshold = 0.3
        self.current_heel_point = None  # Current frame heel point
        self.persistent_heel_point = None  # Persistent heel point
        self.prev_heel_point = None  # Previous frame heel point
        self.heel_crossed_line = False  # Track if heel has crossed the line
        self.heel_returning = False  # Track if heel is returning
        self.prev_heel_side = None  # Previous side of heel point
        self.curr_heel_side = None  # Current side of heel point

        # New toe point tracking variables
        self.current_toe_point = None  # Current frame toe point
        self.persistent_toe_point = None  # Persistent toe point
        self.prev_toe_point = None  # Previous frame toe point
        self.toe_crossed_line = False  # Track if toe has crossed the line
        self.toe_returning = False  # Track if toe is returning
        self.prev_toe_side = None  # Previous side of toe point
        self.curr_toe_side = None  # Current side of toe point

        # Initialize SORTTracker for person tracking
        self.person_tracker = SORTTracker()

        self.yolo_model = YOLO(yolo_model_path)
        self.shoe_model = YOLO(shoe_model_path)
        self.seg_model = SAM(seg_model_path)

        self.yolo_model.to(0)
        self.shoe_model.to(0)
        self.seg_model.to(0)

        self.input_path = input_path
        self.input_type = input_type
        self.frame_files = []
        self.current_frame_idx = 0
        self.skip_seconds = skip_seconds

        if input_type == 'video':
            self.cap = cv2.VideoCapture(input_path)
            if not self.cap.isOpened():
                print("Error: Could not open video.")
                exit()
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:  # frames directory
            if not os.path.isdir(input_path):
                print(f"Error: {input_path} is not a valid directory")
                exit()
            self.frame_files = sorted([f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if not self.frame_files:
                print(f"Error: No image files found in {input_path}")
                exit()
            # Read first frame to get dimensions
            first_frame = cv2.imread(os.path.join(input_path, self.frame_files[0]))
            if first_frame is None:
                print(f"Error: Could not read first frame from {input_path}")
                exit()
            self.frame_height, self.frame_width = first_frame.shape[:2]
            self.fps = fps  # Default FPS for frames

        self.video_name = os.path.basename(input_path).split('.')[0]
        # Create video-specific output directory
        self.output_dir = os.path.join('./misc', self.video_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.centroid_y_values = []
        self.frame_num = 0
        self.prev_time = time.time()

        # Stump rectangle marking variables
        self.stump_rect_start = None
        self.stump_rect_end = None
        self.stump_rect_confirmed = False
        self.stump_rect_drawing = False
        self.stump_rect_box = None  # (x1, y1, x2, y2)

        # Delivery detection/skip logic
        self.delivery_detected_count = 0
        self.skip_frames = 0

        cv2.namedWindow('Video')
        cv2.setMouseCallback('Video', self.select_points)

        self.selected_track_id = None  # Track ID of selected bbox
        self.detected_bboxes = []
        self.detected_track_ids = []

        self.paused = False
        self.pause_frame = None
        self.hoq = False

        if self.right_to_left:
            self.hoq = True

    def get_next_frame(self):

        if self.input_type == 'video':
            ret, frame = self.cap.read()
            if not ret:
                return None
        else:  # frames directory
            if self.current_frame_idx >= len(self.frame_files):
                return None
            frame_path = os.path.join(self.input_path, self.frame_files[self.current_frame_idx])
            frame = cv2.imread(frame_path)
            self.current_frame_idx += 1
        frame = cv2.resize(frame, (1920, 1080))

        if self.hoq:
            self.right_to_left = False
            frame = cv2.flip(frame, 1)
        return frame

    def release_resources(self):
        if self.input_type == 'video':
            self.cap.release()
        cv2.destroyAllWindows()

    def put_text_on_frame(self, frame, text):
        # Define font, text color, and background color
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_color = (255, 255, 255)  # White text
        background_color = (0, 0, 0)  # Black background
        thickness = 2

        # Get the size of the text boxes
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Calculate the background rectangle dimensions
        padding = 5
        rect_width = max(text_width, text_width) + 2 * padding
        rect_height = text_height + text_height + 3 * padding

        # Draw the black rectangle in the top-left corner
        cv2.rectangle(frame, (0, 0), (rect_width, rect_height), background_color, -1)

        # Add the text onto the rectangle
        cv2.putText(frame, text, (padding, text_height + padding), font, font_scale, text_color, thickness)

    def calculate_iou(self, box1, box2):
        if len(box1) != 4 or len(box2) != 4:
            return 0

        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        interArea = max(0, xB - xA + 1) * max(0, yA - yA + 1)

        box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)


        iou = interArea / float(box1Area + box2Area - interArea)

        return iou

    def process_shoe_detections(self, shoe_results, x_min, y_min, frame):
        centroids = []
        # min_centroids = []

        for shoe_result in shoe_results:
            for shoe in shoe_result.boxes:
                sx_min, sy_min, sx_max, sy_max = map(int, shoe.xyxy[0])

                # Adjust shoe bounding box to the original frame coordinates
                sx_min += x_min
                sy_min += y_min
                sx_max += x_min
                sy_max += y_min

                centroid = self.calculate_centroid([sx_min, sy_min, sx_max, sy_max])
                centroids.append(centroid)

                # Find the leftmost shoe
                if centroids:
                    if not self.right_to_left:
                        min_x_point = min(centroids, key=lambda c: c[0])
                        if self.prev_centroid is None or self.prev_centroid[0] > min_x_point[0] and not self.bowler_returning:
                            print('prev_centroid: +++++++++++++++++++++++++++', self.prev_centroid)
                            print('min_x: +++++++++++++++++++++++++++++++++++++++++++++++++', min_x_point)
                            self.min_centroids.append(min_x_point)

                        elif self.bowler_returning:
                            self.min_centroids.clear()
                    else:
                        min_x_point = min(centroids, key=lambda c: c[0])
                        if self.prev_centroid is None or self.prev_centroid[0] < min_x_point[
                            0] and not self.bowler_returning:
                            print('prev_centroid: +++++++++++++++++++++++++++', self.prev_centroid)
                            print('min_x: +++++++++++++++++++++++++++++++++++++++++++++++++', min_x_point)
                            self.min_centroids.append(min_x_point)

                        elif self.bowler_returning:
                            self.min_centroids.clear()

                    self.prev_centroid = min_x_point

                sx_min, sy_min, sx_max, sy_max = self.add_padding_to_bbox([sx_min, sy_min, sx_max, sy_max])
                self.prompted_bbox = [sx_min, sy_min, sx_max, sy_max]  # [x1, y1, x2, y2]


                # # Get the bounding box for the leftmost shoe
                # for shoe in shoe_results[0].boxes:
                #     sx_min, sy_min, sx_max, sy_max = map(int, shoe.xyxy[0])

                #     # Adjust to original frame coordinates
                #     sx_min += x_min
                #     sy_min += y_min
                #     sx_max += x_min
                #     sy_max += y_min

                #     centroid = calculate_centroid([sx_min, sy_min, sx_max, sy_max])

                if centroid == min_x_point:
                    current_shoe_box = [sx_min, sy_min, sx_max, sy_max]

                    # If this is the first frame or no previous shoe is tracked, show the leftmost shoe
                    if self.prev_shoe_box is None:
                        self.prev_shoe_box = current_shoe_box

                    # Calculate IoU to match the current shoe with the previous one
                    iou = self.calculate_iou(self.prev_shoe_box, current_shoe_box)

                    if iou >= self.iou_threshold:
                        # Update the previous shoe bounding box
                        self.prev_shoe_box = current_shoe_box

                    # Draw the bounding box and centroid for the leftmost shoe
                    cv2.rectangle(frame, (sx_min, sy_min), (sx_max, sy_max), (255, 0, 0), 2)
                    cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 5, (0, 0, 255), -1)
                    new_cent = (centroid[0], centroid[1])

        return frame, self.min_centroids
        # return None

    def create_parallel_line_through_point(self, point, line_points):
        """
        Create a parallel line passing through a given point, with the same length as the original line.

        :param point: Tuple of (x, y) representing the point through which the parallel line should pass.
        :param line_points: List of two tuples [(x1, y1), (x2, y2)] representing the original line.
        :return: List of two points representing the new parallel line.
        """
        (x1, y1), (x2, y2) = line_points

        # Find the direction vector of the line
        dx = x2 - x1
        dy = y2 - y1

        # Find the length of the line
        length = np.sqrt(dx ** 2 + dy ** 2)

        # Normalize the direction vector
        if length != 0:
            dx /= length
            dy /= length

        # The new line passes through the given point
        px, py = point
        new_line_point1 = (int(px - dx * length / 2), int(py - dy * length / 2))
        new_line_point2 = (int(px + dx * length / 2), int(py + dy * length / 2))

        return [new_line_point1, new_line_point2]

    def draw_parallel_lines_and_roi(self, image, line_points, point):
        """
        Draws the original line, the parallel line through the given point, and the ROI joining the two lines.

        :param image: The image on which to draw.
        :param line_points: List of two tuples [(x1, y1), (x2, y2)] representing the original line.
        :param point: Tuple of (x, y) representing the point through which the parallel line should pass.
        :return: Image with the drawn lines and ROI.
        """
        # Create the parallel line through the given point
        parallel_line = self.create_parallel_line_through_point(point, line_points)

        # Draw the original line
        cv2.line(image, line_points[0], line_points[1], (0, 255, 0), 2)

        # Draw the parallel line
        cv2.line(image, parallel_line[0], parallel_line[1], (0, 255, 0), 2)

        # Create the ROI by connecting the lines into a quadrilateral
        pts = np.array([line_points[0], line_points[1], parallel_line[1], parallel_line[0]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        return image, pts

    def is_point_in_polygon(self, point, polygon_points):
        """
        Checks if a point lies inside, outside, or on the edge of a polygon.
        """
        # Ensure polygon_points is (N, 2) and int32
        contour = np.array(polygon_points, dtype=np.int32).reshape(-1, 2)
        # Ensure point is a tuple of ints
        point_int = (int(point[0]), int(point[1]))
        result = cv2.pointPolygonTest(contour, point_int, False)
        return result >= 0

    def select_points(self, event, x, y, flags, param):
        # If line selection is not done, handle line points
        if len(self.line_points) < 2:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.line_points.append((x, y))
                if len(self.line_points) == 2:
                    print(f"Selected Line: {self.line_points}")
            return
        # If line is selected but stump not confirmed, handle rectangle
        if not self.stump_rect_confirmed:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.stump_rect_start = (x, y)
                self.stump_rect_end = (x, y)
                self.stump_rect_drawing = True
            elif event == cv2.EVENT_MOUSEMOVE and self.stump_rect_drawing:
                self.stump_rect_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self.stump_rect_drawing:
                self.stump_rect_end = (x, y)
                self.stump_rect_drawing = False
                # If rectangle is drawn and confirmed, set stump_bottom_left and polygon_pts
                if len(self.line_points) == 2:
                    x1, y1 = self.stump_rect_start
                    x2, y2 = self.stump_rect_end
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    self.stump_bottom_left = (x_min, y_max)
                    # Use a dummy image for draw_parallel_lines_and_roi just to get polygon_pts
                    dummy_img = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                    _, self.polygon_pts = self.draw_parallel_lines_and_roi(dummy_img, self.line_points, self.stump_bottom_left)
            # No return here, so we can still use Esc/Enter in run()
            return
        # Only allow bbox selection if line, stump, and polygon_pts are set
        if self.polygon_pts is None:
            return
        # After line and stump are selected, handle bbox selection
        if event == cv2.EVENT_LBUTTONDOWN:
            for idx, (bbox, track_id) in enumerate(zip(self.detected_bboxes, self.detected_track_ids)):
                x1, y1, x2, y2 = bbox
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if self.selected_track_id == track_id:
                        self.selected_track_id = None
                    else:
                        self.selected_track_id = track_id
                    return

    def point_position(self, line, P):
        A, B = self.line_points
        x1, y1 = A
        x2, y2 = B
        x, y = P
        cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        pos = None

        if cross_product > 0:
            pos = "left"
        elif cross_product < 0:
            pos = "right"
        else:
            pos = "on"

        return cross_product, pos

    def point_line_distance(self, point, line):
        x0, y0 = point
        (x1, y1), (x2, y2) = line
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        return numerator / denominator if denominator != 0 else float('inf')

    def find_nearest_point_on_line(self, point, line):
        """Find the nearest point on a line to the given point."""
        x0, y0 = point
        (x1, y1), (x2, y2) = line

        # Vector from line point 1 to point
        dx = x0 - x1
        dy = y0 - y1

        # Vector representing the line
        line_dx = x2 - x1
        line_dy = y2 - y1

        # Line length squared
        line_len_sq = line_dx**2 + line_dy**2

        # Calculate projection ratio (dot product / line length squared)
        if line_len_sq == 0:  # Avoid division by zero
            return (x1, y1)

        ratio = (dx * line_dx + dy * line_dy) / line_len_sq

        # Clamp ratio to [0, 1] to keep point on line segment
        ratio = max(0, min(1, ratio))

        # Calculate nearest point on line
        nearest_x = x1 + ratio * line_dx
        nearest_y = y1 + ratio * line_dy

        return (int(nearest_x), int(nearest_y))

    def detect_peaks(self, y_values, persistence_threshold=3):
        # y_values = list(map(lambda x:x[1], centroids))
        peaks, properties = find_peaks(y_values, prominence=1, distance=persistence_threshold)
        return peaks

    def get_persistent_peak(self, centroids, frame_num):
        global max_y_persistent_peak
        global max_peak_y

        # Initialize max_peak_y if it hasn't been initialized before
        if max_peak_y is None:
            max_peak_y = float('-inf')

        centroid_x_values = list(map(lambda x: x[0], centroids))
        centroid_y_values = list(map(lambda x: x[1], centroids))

        # Detect peaks in the y-values
        peaks = self.detect_peaks(centroid_y_values)

        if centroids and len(self.line_points) == 2:
            if peaks.size > 0:
                # Get the index of the last peak
                max_peak_index = peaks[-1]

                # Get the (x, y) coordinates of the detected peak
                current_peak = (centroid_x_values[max_peak_index], centroid_y_values[max_peak_index])

                # Update max_peak_y if the new peak's y-coordinate is higher
                if centroid_y_values[max_peak_index] > max_peak_y:
                    max_peak_y = centroid_y_values[max_peak_index]
                    max_peak = current_peak
                else:
                    # Use the current highest peak if the detected peak is not higher
                    max_peak = (centroid_x_values[max_peak_index], max_peak_y)

                # Calculate the distance from the peak to the line
                distance_to_line = self.point_line_distance(max_peak, self.line_points)

                # Check if the peak is within the desired range from the line
                if distance_to_line < 50:
                    print("\n=======================================================================================================\n")
                    print(max_peak, max_peak_index)

                    # Update max_y_persistent_peak if it's the first peak or if enough frames have passed
                    if max_y_persistent_peak is None or frame_num - max_y_persistent_peak['frame_num'] >= 3:
                        max_y_persistent_peak = {'peak': max_peak, 'frame_num': frame_num}
                        return max_peak

        return None

    def mark_bowler(self, frame, bbox):
        x_min, y_min, x_max, y_max = bbox

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Crop bowler region
        bowler_region = self.pause_frame[y_min:y_max, x_min:x_max]

        # Detect shoes in the cropped region
        shoe_results = self.shoe_model(bowler_region, conf=0.3, verbose=False, show=False)

        if len(shoe_results) != 0:
            self.shoe_detected = True
        else:
            self.shoe_detected = False
        centroids = []
        centroid_x = (x_min + x_max) / 2
        bottom_center = (centroid_x, y_max)
        self.bowler_bottom_right = (x_max, y_max)
        self.bowler_bottom_left = (x_min, y_max)

        self.bowler_bottom_points = [(int(x), y_max) for x in np.linspace(x_min, x_max, 5)]

        # cross_product, pos = point_position(line_points, bowler_bottom_right)

        if self.bowler_bottom_right and bottom_center and self.bowler_bottom_left:
            if not self.right_to_left:
                if self.prev_bowler_bottom_right and self.prev_bowler_bottom_center and self.prev_bowler_bottom_left:
                    _, self.prev_side = self.point_position(self.line_points, self.prev_bowler_bottom_right)
                    _, self.curr_side = self.point_position(self.line_points, self.bowler_bottom_right)

                    if self.prev_side == 'right' and self.curr_side == 'left' and not self.bowler_crossed_line:
                        self.bowler_crossed_line = True
                        self.put_text_on_frame(frame, '--Bowler crossed line--')
                        print("-------------------------------------------Bowler crossed line-------------------------------------------")
                        # cv2.imwrite('/home/soumyadeep@quidich.local/soumyadeep/No_Ball/bowled.jpg', frame)

                    if self.prev_bowler_bottom_center[0] < bottom_center[0] and self.prev_bowler_bottom_left[0] < self.bowler_bottom_left[0] and self.prev_bowler_bottom_right[0] < self.bowler_bottom_right[0]:
                        self.bowler_returning = True
                        self.put_text_on_frame(frame, f'--Bowler returning-- {self.prev_bowler_bottom_center[0], bottom_center[0], self.prev_bowler_bottom_left[0], self.bowler_bottom_left[0], self.prev_bowler_bottom_right[0], self.bowler_bottom_right[0]}')
                        print("-------------------------------------------Bowler returning-------------------------------------------")
                        # cv2.imwrite('/home/soumyadeep@quidich.local/soumyadeep/No_Ball/bowler_returning.jpg', frame)

                    elif self.prev_bowler_bottom_center[0] > bottom_center[0] and self.prev_bowler_bottom_left[0] > self.bowler_bottom_left[0] and self.prev_bowler_bottom_right[0] > self.bowler_bottom_right[0]:
                        self.bowler_returning = False

            else:
                if self.prev_bowler_bottom_right and self.prev_bowler_bottom_center and self.prev_bowler_bottom_left:
                    _, self.prev_side = self.point_position(self.line_points, self.prev_bowler_bottom_right)
                    _, self.curr_side = self.point_position(self.line_points, self.bowler_bottom_right)

                    if self.prev_side == 'left' and self.curr_side == 'right' and not self.bowler_crossed_line:
                        self.bowler_crossed_line = True
                        self.put_text_on_frame(frame, '--Bowler crossed line--')
                        print(
                            "-------------------------------------------Bowler crossed line-------------------------------------------")
                        # cv2.imwrite('/home/soumyadeep@quidich.local/soumyadeep/No_Ball/bowled.jpg', frame)

                    if self.prev_bowler_bottom_center[0] > bottom_center[0] and self.prev_bowler_bottom_left[0] > \
                            self.bowler_bottom_left[0] and self.prev_bowler_bottom_right[0] > self.bowler_bottom_right[
                        0]:
                        self.bowler_returning = True
                        self.put_text_on_frame(frame,
                                               f'--Bowler returning-- {self.prev_bowler_bottom_center[0], bottom_center[0], self.prev_bowler_bottom_left[0], self.bowler_bottom_left[0], self.prev_bowler_bottom_right[0], self.bowler_bottom_right[0]}')
                        print(
                            "-------------------------------------------Bowler returning-------------------------------------------")
                        # cv2.imwrite('/home/soumyadeep@quidich.local/soumyadeep/No_Ball/bowler_returning.jpg', frame)

                    elif self.prev_bowler_bottom_center[0] < bottom_center[0] and self.prev_bowler_bottom_left[0] < \
                            self.bowler_bottom_left[0] and self.prev_bowler_bottom_right[0] < self.bowler_bottom_right[
                        0]:
                        self.bowler_returning = False

            self.prev_bowler_bottom_right = self.bowler_bottom_right
            self.prev_bowler_bottom_center = bottom_center
            self.prev_bowler_bottom_left = self.bowler_bottom_left

        if self.is_point_in_polygon(bottom_center, self.polygon_pts):
            cv2.circle(frame, (int(centroid_x), int(y_max)), 4, (0, 0, 255), -1)
            frame, self.min_centroids = self.process_shoe_detections(shoe_results, x_min, y_min, frame)

        return frame

    def find_persistent_max_y(self, centroids, frame_num, persistence_threshold=3):
        check_for_foot = False

        if centroids and len(self.line_points) == 2:
            # sorted_centroids = sorted(centroids, key=lambda c: c[1], reverse=True)
            sorted_centroids = sorted(centroids, key=lambda c: (-c[1], c[0]))  #sort with max_y them min_x
            for centroid in sorted_centroids:
                distance_to_line = self.point_line_distance(centroid, self.line_points)
                if distance_to_line < 100:
                    check_for_foot = True
                elif self.is_point_in_polygon(centroid, self.polygon_pts):
                    check_for_foot = True
                else:
                    check_for_foot = False

                if check_for_foot:
                    if self.max_y_persistent_centroid is None or frame_num - self.max_y_persistent_centroid['frame_num'] >= persistence_threshold:
                        self.max_y_persistent_centroid = {'centroid': centroid, 'frame_num': frame_num}
                        return centroid
        return None

    def calculate_centroid(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        centroid_x = (x_min + x_max) / 2
        centroid_y = (y_min + y_max) / 2
        return [centroid_x, centroid_y]

    def add_padding_to_bbox(self, bbox, padding_factor=0.1):
        x_min, y_min, x_max, y_max = bbox
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        bbox_padded_width = bbox_width * padding_factor
        bbox_padded_height = bbox_height * padding_factor
        bbox_padded_x_min = int(x_min - (bbox_padded_width / 2))
        bbox_padded_y_min = int(y_min - (bbox_padded_height / 2))
        bbox_padded_x_max = int(x_max + (bbox_padded_width / 2))
        bbox_padded_y_max = int(y_max + (bbox_padded_height / 2))
        if bbox_padded_x_min < 0:
            bbox_padded_x_min = 0
        if bbox_padded_y_min < 0:
            bbox_padded_y_min = 0
        if bbox_padded_x_max > self.frame_width:
            bbox_padded_x_max = self.frame_width
        if bbox_padded_y_max > self.frame_height:
            bbox_padded_y_max = self.frame_height
        return [bbox_padded_x_min, bbox_padded_y_min, bbox_padded_x_max, bbox_padded_y_max]

    def check_no_ball(self, prompted_bbox, heel_point, toe_point=None):
        crease_line = np.asarray(self.line_points)
        crease_line_x = np.min(crease_line[:,0])
        center_point_x, center_point_y = (prompted_bbox[0] + prompted_bbox[2]) // 2, (prompted_bbox[1] + prompted_bbox[3]) // 2

        # Check no ball using heel point
        if heel_point:
            heel_point_x, heel_point_y = heel_point
            if center_point_x < crease_line_x and heel_point_x < crease_line_x:
                print("-"*50)
                print("it's no ball (heel)")
                print("-"*50)

        # Check no ball using toe point
        if toe_point:
            toe_point_x, toe_point_y = toe_point
            if center_point_x < crease_line_x and toe_point_x < crease_line_x:
                print("-"*50)
                print("it's no ball (toe)")
                print("-"*50)

        # Original commented code
        # else:
        #     if center_point_x < crease_line_x and prompted_bbox[2] < crease_line_x:
        #         print("-"*50)
        #         print("it's no ball")
        #         print("-"*50)

    def calculate_angle_with_horizontal(self, heel_point, toe_point):
        """
        Calculate the angle between the toe-heel line and a horizontal line.
        
        Args:
            heel_point: Tuple (x, y) representing the heel point
            toe_point: Tuple (x, y) representing the toe point
            
        Returns:
            float: Angle in degrees between the toe-heel line and horizontal (0-360)
        """
        # Create a horizontal line at the same y-coordinate as the heel point
        horizontal_line = [(0, heel_point[1]), (self.frame_width, heel_point[1])]
        toe_heel_line = [heel_point, toe_point]
        
        # Calculate slopes
        (x1, y1), (x2, y2) = toe_heel_line
        (x3, y3), (x4, y4) = horizontal_line
        
        # Calculate slopes
        slope1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
        slope2 = (y4 - y3) / (x4 - x3) if (x4 - x3) != 0 else float('inf')
        
        # Calculate angle in radians
        if slope1 == float('inf'):
            angle_rad = np.pi/2  # 90 degrees
        else:
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
        
        # Convert to degrees and ensure positive angle
        angle_deg = np.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
            
        return angle_deg

    def calculate_angle_between_lines(self, line1_points, line2_points):
        """
        Calculate the angle between two lines in degrees.
        
        Args:
            line1_points: List of two points [(x1, y1), (x2, y2)] representing the first line
            line2_points: List of two points [(x1, y1), (x2, y2)] representing the second line
            
        Returns:
            float: Angle between the two lines in degrees (0-180)
        """
        # Extract points for both lines
        (x1, y1), (x2, y2) = line1_points
        (x3, y3), (x4, y4) = line2_points
        
        # Calculate slopes of both lines
        slope1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
        slope2 = (y4 - y3) / (x4 - x3) if (x4 - x3) != 0 else float('inf')
        
        # Handle vertical lines (infinite slope)
        if slope1 == float('inf') and slope2 == float('inf'):
            return 0.0  # Both lines are vertical and parallel
        elif slope1 == float('inf'):
            # First line is vertical, second line has slope slope2
            angle = 90 - np.degrees(np.arctan(abs(slope2)))
        elif slope2 == float('inf'):
            # Second line is vertical, first line has slope slope1
            angle = 90 - np.degrees(np.arctan(abs(slope1)))
        else:
            # Calculate angle between two non-vertical lines
            angle = np.degrees(np.arctan(abs((slope2 - slope1) / (1 + slope1 * slope2))))
        
        # Ensure angle is between 0 and 180 degrees
        return min(angle, 180 - angle)

    def draw_segmentation(self, shoe_seg, prompted_bbox, frame):
        masks = shoe_seg[0].masks.data.cpu().numpy()
        heel_point = None
        toe_point = None
        persistent_centroid_x, persistent_centroid_y = int(self.persistent_centroid[0]), int(
            self.persistent_centroid[1])
        # Create an empty image for the mask
        mask_image = np.zeros_like(frame)

        for mask in masks:
            mask = mask.astype(np.uint8)
            resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            bbox_mask = np.zeros_like(resized_mask)

            # Apply mask only within the bounding box
            bbox_mask[prompted_bbox[1]:prompted_bbox[3], prompted_bbox[0]:prompted_bbox[2]] = resized_mask[prompted_bbox[1]:prompted_bbox[3], prompted_bbox[0]:prompted_bbox[2]]

            # Find points in the segmentation mask
            if np.any(bbox_mask):
                y_coords, x_coords = np.where(bbox_mask == 1)
                if len(x_coords) > 0:
                    # Find heel point (maximum y coordinate - lowest point in the foot)
                    max_y_idx = np.argmax(y_coords)
                    heel_point = (x_coords[max_y_idx], y_coords[max_y_idx])

                    # Find toe point based on direction (either rightmost or leftmost point)
                    if not self.right_to_left:
                        # For left-to-right movement, toe is the minimum x coordinate
                        min_x_idx = np.argmin(x_coords)
                        toe_point = (x_coords[min_x_idx], y_coords[min_x_idx])

                        # Handle special case for heel
                        if heel_point[0] <= persistent_centroid_x:
                            max_x_idx = np.argmax(x_coords)
                            heel_point = (x_coords[max_x_idx], y_coords[max_x_idx])
                    else:
                        # For right-to-left movement, toe is the maximum x coordinate
                        max_x_idx = np.argmax(x_coords)
                        toe_point = (x_coords[max_x_idx], y_coords[max_x_idx])

                        # Handle special case for heel
                        if heel_point[0] >= persistent_centroid_x:
                            min_x_idx = np.argmin(x_coords)
                            heel_point = (x_coords[min_x_idx], y_coords[min_x_idx])

            mask_image[bbox_mask == 1] = [0, 255, 0]  # Color the mask (green)

        segmented_image = cv2.addWeighted(frame, 0.7, mask_image, 0.3, 0)

        # Save the current heel point for this frame
        self.current_heel_point = heel_point
        # Save the current toe point for this frame
        self.current_toe_point = toe_point

        # Check for no ball using both points
        self.check_no_ball(prompted_bbox, heel_point, toe_point)

        # Process heel point
        if heel_point:
            if self.persistent_heel_point is None:
                self.persistent_heel_point = heel_point
                self.heel_points.append(heel_point)

            # Draw current heel point
            cv2.circle(segmented_image, heel_point, 5, (0, 0, 255), -1)  # Red dot for heel
            cv2.putText(segmented_image, 'Heel Point', (heel_point[0] - 30, heel_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Calculate and visualize the distance from the heel point to the selected line
            if len(self.line_points) == 2:
                # Find distance from heel point to line
                distance = self.point_line_distance(heel_point, self.line_points)

                # Find nearest point on line
                nearest_point = self.find_nearest_point_on_line(heel_point, self.line_points)

                # Draw line from heel point to nearest point on line
                cv2.line(segmented_image, heel_point, nearest_point, (255, 0, 255), 2)

                # Add the distance text
                cv2.putText(segmented_image, f'Heel Dist: {distance:.2f} px',
                           (heel_point[0] + 10, heel_point[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                # Store the heel point if it's closest to the line
                if self.heel_point is None or distance < self.point_line_distance(self.heel_point, self.line_points):
                    self.heel_point = heel_point
                    self.heel_points.append(heel_point)
                    # Update persistent heel point if this one is closer to the line
                    self.persistent_heel_point = heel_point

                # Track heel point movement and line crossing
                if self.prev_heel_point is not None:
                    _, self.prev_heel_side = self.point_position(self.line_points, self.prev_heel_point)
                    _, self.curr_heel_side = self.point_position(self.line_points, heel_point)

                    if not self.right_to_left:
                        if self.prev_heel_side == 'right' and self.curr_heel_side == 'left' and not self.heel_crossed_line:
                            self.heel_crossed_line = True
                            self.put_text_on_frame(segmented_image, '--Heel crossed line--')
                            print("-------------------------------------------Heel crossed line-------------------------------------------")
                            # cv2.imwrite(os.path.join(self.output_dir, f'heel_crossed_line_{self.frame_num}.jpg'), segmented_image)

                        if self.prev_heel_point[0] < heel_point[0] and not self.heel_crossed_line:
                            self.heel_returning = True
                            self.put_text_on_frame(segmented_image, f'--Heel returning-- {self.prev_heel_point[0], heel_point[0]}')
                            print("-------------------------------------------Heel returning-------------------------------------------")
                            # cv2.imwrite(os.path.join(self.output_dir, f'heel_returning_{self.frame_num}.jpg'), segmented_image)
                        elif self.prev_heel_point[0] > heel_point[0]:
                            self.heel_returning = False
                    else:
                        if self.prev_heel_side == 'left' and self.curr_heel_side == 'right' and not self.heel_crossed_line:
                            self.heel_crossed_line = True
                            self.put_text_on_frame(segmented_image, '--Heel crossed line--')
                            print("-------------------------------------------Heel crossed line-------------------------------------------")
                            # cv2.imwrite(os.path.join(self.output_dir, f'heel_crossed_line_{self.frame_num}.jpg'), segmented_image)

                        if self.prev_heel_point[0] > heel_point[0] and not self.heel_crossed_line:
                            self.heel_returning = True
                            self.put_text_on_frame(segmented_image, f'--Heel returning-- {self.prev_heel_point[0], heel_point[0]}')
                            print("-------------------------------------------Heel returning-------------------------------------------")
                            # cv2.imwrite(os.path.join(self.output_dir, f'heel_returning_{self.frame_num}.jpg'), segmented_image)
                        elif self.prev_heel_point[0] < heel_point[0]:
                            self.heel_returning = False

                self.prev_heel_point = heel_point

        # Process toe point
        if toe_point:
            if self.persistent_toe_point is None:
                self.persistent_toe_point = toe_point
                self.toe_points.append(toe_point)

            # Draw current toe point
            cv2.circle(segmented_image, toe_point, 5, (255, 0, 0), -1)  # Blue dot for toe
            cv2.putText(segmented_image, 'Toe Point', (toe_point[0] - 30, toe_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Calculate and visualize the distance from the toe point to the selected line
            if len(self.line_points) == 2:
                # Find distance from toe point to line
                toe_distance = self.point_line_distance(toe_point, self.line_points)

                # Find nearest point on line
                toe_nearest_point = self.find_nearest_point_on_line(toe_point, self.line_points)

                # Draw line from toe point to nearest point on line
                cv2.line(segmented_image, toe_point, toe_nearest_point, (0, 255, 255), 2)

                # Add the distance text
                cv2.putText(segmented_image, f'Toe Dist: {toe_distance:.2f} px',
                           (toe_point[0] + 10, toe_point[1] + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Store the toe point if it's closest to the line
                if self.toe_point is None or toe_distance < self.point_line_distance(self.toe_point, self.line_points):
                    self.toe_point = toe_point
                    self.toe_points.append(toe_point)
                    # Update persistent toe point if this one is closer to the line
                    self.persistent_toe_point = toe_point

                # Track toe point movement and line crossing
                if self.prev_toe_point is not None:
                    _, self.prev_toe_side = self.point_position(self.line_points, self.prev_toe_point)
                    _, self.curr_toe_side = self.point_position(self.line_points, toe_point)

                    if not self.right_to_left:
                        if self.prev_toe_side == 'right' and self.curr_toe_side == 'left' and not self.toe_crossed_line:
                            self.toe_crossed_line = True
                            self.put_text_on_frame(segmented_image, '--Toe crossed line--')
                            print("-------------------------------------------Toe crossed line-------------------------------------------")
                            # cv2.imwrite(os.path.join(self.output_dir, f'toe_crossed_line_{self.frame_num}.jpg'), segmented_image)

                        if self.prev_toe_point[0] < toe_point[0] and not self.toe_crossed_line:
                            self.toe_returning = True
                            self.put_text_on_frame(segmented_image, f'--Toe returning-- {self.prev_toe_point[0], toe_point[0]}')
                            print("-------------------------------------------Toe returning-------------------------------------------")
                            # cv2.imwrite(os.path.join(self.output_dir, f'toe_returning_{self.frame_num}.jpg'), segmented_image)
                        elif self.prev_toe_point[0] > toe_point[0]:
                            self.toe_returning = False
                    else:
                        if self.prev_toe_side == 'left' and self.curr_toe_side == 'right' and not self.toe_crossed_line:
                            self.toe_crossed_line = True
                            self.put_text_on_frame(segmented_image, '--Toe crossed line--')
                            print("-------------------------------------------Toe crossed line-------------------------------------------")
                            # cv2.imwrite(os.path.join(self.output_dir, f'toe_crossed_line_{self.frame_num}.jpg'), segmented_image)

                        if self.prev_toe_point[0] > toe_point[0] and not self.toe_crossed_line:
                            self.toe_returning = True
                            self.put_text_on_frame(segmented_image, f'--Toe returning-- {self.prev_toe_point[0], toe_point[0]}')
                            print("-------------------------------------------Toe returning-------------------------------------------")
                            # cv2.imwrite(os.path.join(self.output_dir, f'toe_returning_{self.frame_num}.jpg'), segmented_image)
                        elif self.prev_toe_point[0] < toe_point[0]:
                            self.toe_returning = False

                self.prev_toe_point = toe_point

        # If both heel and toe are detected, draw a line between them and calculate angle
        if heel_point and toe_point:
            # Calculate distance between heel and toe points
            distance = np.sqrt((toe_point[0] - heel_point[0])**2 + (toe_point[1] - heel_point[1])**2)

            # Only proceed if distance is greater than 50 pixels
            if distance > 50:
                # Draw line between heel and toe
                cv2.line(segmented_image, heel_point, toe_point, (0, 165, 255), 2)  # Orange line between heel and toe

                # Calculate angle with horizontal axis
                # Create a horizontal line (x-axis) at y=0
                x_axis_line = [(0, 0), (frame.shape[1], 0)]
                toe_heel_line = [toe_point, heel_point]
                angle = self.calculate_angle_between_lines(toe_heel_line, x_axis_line)

                # Check if angle indicates ground impact (160-220 degrees)
                if -10 <= angle <= 10:
                    impact_text = "Ground Impact Detected!"
                    impact_color = (0, 255, 0)  # Green color for impact

                    # Check if both heel and toe points have crossed the crease line
                    _, heel_side = self.point_position(self.line_points, heel_point)
                    _, toe_side = self.point_position(self.line_points, toe_point)

                    # Calculate heel point distance from the selected line
                    heel_distance = self.point_line_distance(heel_point, self.line_points)

                    if heel_side == 'left' and toe_side == 'left' and heel_distance > 5:
                        no_ball_text = "NO BALL - Both points crossed crease"
                        no_ball_color = (0, 0, 255)  # Red color for no ball
                        print("-------------------------------------------NO BALL DETECTED-------------------------------------------")
                        cv2.imwrite(os.path.join(self.output_dir, f'no_ball_detected_{self.frame_num}.jpg'), segmented_image)
                    else:
                        no_ball_text = "Normal Delivery"
                        no_ball_color = (0, 255, 0)  # Green color for normal delivery
                    
                    self.delivery_detected_count += 1
                    if self.delivery_detected_count >= 5:
                        self.skip_frames = int(self.skip_seconds * self.fps)
                        self.delivery_detected_count = 0
                    
                else:
                    impact_text = "No Ground Impact"
                    impact_color = (0, 0, 255)  # Red color for no impact
                    no_ball_text = ""
                    no_ball_color = (0, 0, 0)

                # Display angle and impact information
                angle_text = f"Angle: {angle:.1f}°"
                cv2.putText(segmented_image, angle_text, (heel_point[0], heel_point[1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(segmented_image, impact_text, (heel_point[0], heel_point[1] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, impact_color, 2)

                # Display no ball status if ground impact is detected
                if no_ball_text:
                    cv2.putText(segmented_image, no_ball_text, (heel_point[0], heel_point[1] - 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, no_ball_color, 2)

                # Draw circles for heel and toe points
                cv2.circle(segmented_image, heel_point, 5, (0, 0, 255), -1)  # Red dot for heel
                cv2.circle(segmented_image, toe_point, 5, (255, 0, 0), -1)  # Blue dot for toe

                # Add labels for heel and toe points
                cv2.putText(segmented_image, 'Heel Point', (heel_point[0] - 30, heel_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(segmented_image, 'Toe Point', (toe_point[0] - 30, toe_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


        return segmented_image, heel_point, toe_point

    def calculate_performance_metrics(self):
        """Calculate CPU, GPU, and memory usage metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        
        # GPU memory usage (if available)
        gpu_memory_mb = 0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
        
        # Memory usage in MB
        memory = psutil.virtual_memory()
        memory_used_mb = (memory.total - memory.available) / (1024 * 1024)  # Convert to MB
        memory_total_mb = memory.total / (1024 * 1024)  # Convert to MB
        
        return cpu_percent, gpu_memory_mb, memory_used_mb, memory_total_mb

    def put_performance_metrics(self, frame, process_time, cpu_percent, gpu_memory_mb, memory_used_mb, memory_total_mb):
        """Add performance metrics to the frame."""
        # Define font and colors
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)  # Green color
        thickness = 2
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Create text lines
        lines = [
            f"Process Time: {process_time:.2f} ms",
            f"CPU Usage: {cpu_percent:.1f}%",
            f"GPU Memory: {gpu_memory_mb:.1f} MB",
            f"RAM: {memory_used_mb:.1f}/{memory_total_mb:.1f} MB"
        ]
        
        # Calculate maximum text width
        max_text_width = 0
        for line in lines:
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_text_width = max(max_text_width, text_width)
        
        # Add padding to the right
        padding = 10
        x_position = width - max_text_width - padding
        
        # Add each line to the frame starting from top
        for i, line in enumerate(lines):
            y = 20 * (i + 1)  # Start from top with 20px spacing
            cv2.putText(frame, line, (x_position, y), font, font_scale, color, thickness)
        
        return frame

    def run(self):
        logger.info("Starting no-ball detection process")
        # --- Wait for line selection on the first frame ---
        first_frame = self.get_next_frame()
        if first_frame is None:
            logger.info("End of input or error reading frame.")
            return
        selecting = True
        while selecting:
            frame_copy = first_frame.copy()
            if len(self.line_points) < 2:
                self.put_text_on_frame(frame_copy, "Select 2 points for the crease line (Left click). Press Enter to confirm, Esc to reset.")
            else:
                self.put_text_on_frame(frame_copy, "Press Enter to confirm points, Esc to reset.")
            if len(self.line_points) >= 1:
                for pt in self.line_points:
                    cv2.circle(frame_copy, pt, 5, (0, 255, 255), -1)
            cv2.imshow('Video', frame_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Process terminated by user during line selection")
                self.release_resources()
                return
            elif key in [13, 10]:  # Enter key (13 on Windows, 10 on some systems)
                if len(self.line_points) == 2:
                    selecting = False
            elif key == 27:  # Escape key
                self.line_points.clear()
        logger.info(f"Line selected: {self.line_points}")
        # --- End of line selection logic ---

        # --- Stump rectangle marking logic ---
        stump_selecting = True
        while stump_selecting:
            frame_copy = first_frame.copy()
            # Draw crease line
            if len(self.line_points) == 2:
                cv2.line(frame_copy, self.line_points[0], self.line_points[1], (0, 255, 255), 2)
            # Draw rectangle if being drawn or finished
            if self.stump_rect_start and self.stump_rect_end:
                x1, y1 = self.stump_rect_start
                x2, y2 = self.stump_rect_end
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if not self.stump_rect_confirmed:
                self.put_text_on_frame(frame_copy, "Mark stumps: Drag to draw rectangle. Enter to confirm, Esc to reset.")
            else:
                self.put_text_on_frame(frame_copy, "Stump marked. Press Enter to continue, Esc to reset.")
            cv2.imshow('Video', frame_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Process terminated by user during stump marking")
                self.release_resources()
                return
            elif key in [13, 10]:  # Enter
                if self.stump_rect_start and self.stump_rect_end and not self.stump_rect_confirmed:
                    # Confirm rectangle
                    x1, y1 = self.stump_rect_start
                    x2, y2 = self.stump_rect_end
                    # Normalize coordinates
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    self.stump_rect_box = (x_min, y_min, x_max, y_max)
                    self.stump_rect_confirmed = True
                elif self.stump_rect_confirmed:
                    stump_selecting = False
            elif key == 27:  # Escape
                self.stump_rect_start = None
                self.stump_rect_end = None
                self.stump_rect_confirmed = False
                self.stump_rect_box = None
        logger.info(f"Stump rectangle selected: {self.stump_rect_box}")
        # --- End of stump rectangle marking logic ---

        # Reset video/frame index to start from the beginning
        if self.input_type == 'video':
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            self.current_frame_idx = 0
        self.frame_num = 0
        self.prev_time = time.time()
        # --- Main processing loop ---
        while True:
            frame_start_time = time.time()
            if not self.paused or self.pause_frame is None:
                frame = self.get_next_frame()
                if frame is None:
                    logger.info("End of input or error reading frame.")
                    break
                self.pause_frame = frame.copy()
            else:
                frame = self.pause_frame.copy()

            # --- DETECTION AND BBOX SELECTION ---
            self.detected_bboxes = []
            self.detected_track_ids = []
            yolo_results = self.yolo_model(frame, verbose=False, show=False)
            # Collect all person (bowler) detections
            person_detections = []
            for result in yolo_results:
                for box in result.boxes:
                    if box.cls == 0 and box.conf >= 0.3:
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                        x_min, y_min, x_max, y_max = self.add_padding_to_bbox([x_min, y_min, x_max, y_max])
                        conf = float(box.conf)
                        person_detections.append([x_min, y_min, x_max, y_max, conf, 0])  # 0 for class_id (person)
            # Convert to supervision Detections for tracker
            if person_detections:
                dets_np = np.array(person_detections)
                detections = sv.Detections(
                    xyxy=dets_np[:, :4],
                    confidence=dets_np[:, 4],
                    class_id=dets_np[:, 5].astype(int)
                )
                tracked = self.person_tracker.update(detections)
                # tracked.xyxy, tracked.tracker_id
                for xyxy, track_id in zip(tracked.xyxy.astype(int), tracked.tracker_id):
                    x1, y1, x2, y2 = xyxy
                    self.detected_bboxes.append([x1, y1, x2, y2])
                    self.detected_track_ids.append(track_id)
            # Draw all detected bboxes with track IDs
            found_selected = False
            for idx, (bbox, track_id) in enumerate(zip(self.detected_bboxes, self.detected_track_ids)):
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0)  # Green for unselected
                thickness = 2
                if self.selected_track_id == track_id:
                    color = (0, 0, 255)  # Red for selected
                    thickness = 3
                    found_selected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                if self.selected_track_id == track_id:
                    cv2.putText(frame, 'SELECTED', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                # Display track ID above each bbox
                cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            # If selected_track_id is set but not found in this frame, auto-unselect
            if self.selected_track_id is not None and not found_selected:
                self.selected_track_id = None

            # Find the bbox for the selected track_id (if any)
            selected_bbox = None
            if self.selected_track_id is not None:
                for bbox, track_id in zip(self.detected_bboxes, self.detected_track_ids):
                    if track_id == self.selected_track_id:
                        selected_bbox = bbox
                        break

            # --- Visualization: Draw crease line, stumps, and ROI ---
            if len(self.line_points) == 2 and self.stump_rect_box is not None and self.stump_bottom_left is not None:
                # Draw crease line
                cv2.line(frame, self.line_points[0], self.line_points[1], (0, 255, 255), 3)
                # Draw stumps rectangle
                x1, y1, x2, y2 = self.stump_rect_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # Draw ROI (polygon between crease and stumps)
                frame, _ = self.draw_parallel_lines_and_roi(frame, self.line_points, self.stump_bottom_left)

            # Show PAUSED overlay if paused
            if self.paused:
                cv2.putText(frame, 'PAUSED', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

            # Only process bowler, shoe, segmentation, etc. if a bbox is selected
            if selected_bbox is not None and self.polygon_pts is not None:
                # Mark bowler using the selected bbox
                frame = self.mark_bowler(frame, selected_bbox)
                # After detection loop, always use user-marked stumps
                if self.stump_rect_box:
                    x1, y1, x2, y2 = self.stump_rect_box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    self.stump_bottom_left = (x1, y2)
                if len(self.line_points) == 2:
                    _, self.polygon_pts = self.draw_parallel_lines_and_roi(frame, self.line_points, self.stump_bottom_left)
                # Only process shoe/segmentation for selected bbox
                bowler_detected = True
                segmented_image = None
                if bowler_detected and not self.bowler_returning:
                    if self.shoe_detected:
                        self.persistent_centroid = self.find_persistent_max_y(self.min_centroids, self.frame_num)
                        if self.persistent_centroid:
                            px, py = self.persistent_centroid
                            if (self.prompted_bbox[0] <= px <= self.prompted_bbox[2] and
                                self.prompted_bbox[1] <= py <= self.prompted_bbox[3]):
                                self.persistent_counter += 1
                                cv2.circle(frame, (int(self.persistent_centroid[0]), int(self.persistent_centroid[1])), 10, (255, 0, 0), -1)
                                cv2.putText(frame, f'Persistent Max Y', (int(self.persistent_centroid[0]), int(self.persistent_centroid[1]) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                if self.persistent_counter > 1:  # Use 1 instead of fps to avoid NameError
                                    self.persistent_counter = 0
                                seg_results = self.seg_model(source=self.pause_frame, points=[self.persistent_centroid], conf=0.7)
                                segmented_image, current_heel_point, current_toe_point = self.draw_segmentation(seg_results, self.prompted_bbox, frame)
                            else:
                                self.persistent_centroid = None
                                self.persistent_counter = 0
                    else:
                        self.persistent_centroid = None
                        self.persistent_counter = 0
                # Log performance metrics
                process_time = (time.time() - frame_start_time) * 1000  # Convert to milliseconds
                cpu_percent, gpu_memory_mb, memory_used_mb, memory_total_mb = self.calculate_performance_metrics()
                frame = self.put_performance_metrics(frame, process_time, cpu_percent, gpu_memory_mb, memory_used_mb, memory_total_mb)
                cv2.imshow('Video', frame)
                if segmented_image is not None:
                    segmented_image = self.put_performance_metrics(segmented_image, process_time, cpu_percent, gpu_memory_mb, memory_used_mb, memory_total_mb)
                    cv2.imshow('Video', segmented_image)
                    cv2.imwrite(os.path.join(self.output_dir, f'foot_seg_frame{self.frame_num}.jpg'), segmented_image)
                self.frame_num += 1
                key = cv2.waitKey(0 if self.paused else 1) & 0xFF
                if key == ord('q'):
                    logger.info("Process terminated by user")
                    break
                elif key == ord(' '):  # Spacebar to toggle pause
                    self.paused = not self.paused
                elif self.paused and key == ord('a'):
                    # Go back one frame
                    if self.input_type == 'video':
                        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 2))
                        self.pause_frame = self.get_next_frame()
                    else:
                        self.current_frame_idx = max(0, self.current_frame_idx - 2)
                        self.pause_frame = self.get_next_frame()
                elif self.paused and key == ord('d'):
                    # Go forward one frame
                    self.pause_frame = self.get_next_frame()
                continue  # Skip the rest of the loop if bbox is selected

            # If no bbox is selected, just show the UI and frame
            process_time = (time.time() - frame_start_time) * 1000  # Convert to milliseconds
            cpu_percent, gpu_memory_mb, memory_used_mb, memory_total_mb = self.calculate_performance_metrics()
            frame = self.put_performance_metrics(frame, process_time, cpu_percent, gpu_memory_mb, memory_used_mb, memory_total_mb)
            cv2.imshow('Video', frame)
            self.frame_num += 1
            key = cv2.waitKey(0 if self.paused else 1) & 0xFF
            if key == ord('q'):
                logger.info("Process terminated by user")
                break
            elif key == ord(' '):  # Spacebar to toggle pause
                self.paused = not self.paused
            elif self.paused and key == ord('a'):
                # Go back one frame
                if self.input_type == 'video':
                    pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 2))
                    self.pause_frame = self.get_next_frame()
                else:
                    self.current_frame_idx = max(0, self.current_frame_idx - 2)
                    self.pause_frame = self.get_next_frame()
            elif self.paused and key == ord('d'):
                # Go forward one frame
                self.pause_frame = self.get_next_frame()

        self.release_resources()
        logger.info("Process completed successfully")





if __name__ == "__main__":
    # Example usage for video
    detector = NoBallDetector(
        yolo_model_path="./models/yolo11l.pt",
        shoe_model_path="./models/shoe_det_best_v1.pt",
        seg_model_path="./models/sam2.1_l.pt",
        # input_path="E:/amnt/quidich/data/Test_videos/SHGN1_S001_S002_T238_deinterlaced.mp4",
        # input_path="E:/amnt/quidich/data/17apr/camera08/22_39_17apr25_exp64.mp4",
        input_path="E:/amnt/quidich/data/Test_videos/IND_BAN_TEST_1.MOV",
        input_type='video',
        right_to_left=True,
        skip_seconds=10
    )
    detector.run()

    # # Example usage for frames directory
    # detector = NoBallDetector(
    #     yolo_model_path="./models/yolo11l.pt",
    #     shoe_model_path="./models/shoe_det_best_v1.pt",
    #     seg_model_path="./models/sam2.1_l.pt",
    #     input_path="../data/17apr/camera08/23_47_17apr25_exp73",
    #     input_type='frames',
    #     right_to_left=False,
    #     fps=100,
    #     skip_seconds=10
    # )
    # detector.run()
