import torch
from ultralytics import YOLO, SAM
import cv2
import numpy as np
import time
# import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from shapely.geometry import Point, Polygon

class NoBallDetector:
    def __init__(self, bowler_model_path, shoe_model_path, seg_model_path, video_path):
        self.line_points = []
        self.max_y_persistent_peak = None
        self.max_y_persistent_centroid = None
        self.persistent_centroid = None
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

        self.bowler_model = YOLO(bowler_model_path)
        self.shoe_model = YOLO(shoe_model_path)
        self.seg_model = SAM(seg_model_path)

        self.bowler_model.to(0)
        self.shoe_model.to(0)
        self.seg_model.to(0)

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.video_name = video_path.split('/')[-1].split('.')[0]

        if not self.cap.isOpened():
            print("Error: Could not open video.")
            exit()

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.centroid_y_values = []
        self.frame_num = 0
        self.prev_time = time.time()

        cv2.namedWindow('Video')
        cv2.setMouseCallback('Video', self.select_points)

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
                    min_x_point = min(centroids, key=lambda c: c[0])
                    if self.prev_centroid is None or self.prev_centroid[0] > min_x_point[0] and not self.bowler_returning:
                        print('prev_centroid: +++++++++++++++++++++++++++', self.prev_centroid)                    
                        print('min_x: +++++++++++++++++++++++++++++++++++++++++++++++++', min_x_point)
                        self.min_centroids.append(min_x_point)

                    elif self.bowler_returning:
                        self.min_centroids.clear()

                    self.prev_centroid = min_x_point
                
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

        :param point: Tuple (x, y) representing the point to check.
        :param polygon_points: List of tuples [(x1, y1), (x2, y2), ...] representing the polygon vertices.
        :return: Boolean indicating if the point lies inside the polygon.
        """
        # Convert the list of points to a format compatible with cv2.pointPolygonTest
        contour = np.array(polygon_points, dtype=np.int32)
        
        # Use cv2.pointPolygonTest to check if the point is inside (-1 for outside, 0 for on the edge, 1 for inside)
        result = cv2.pointPolygonTest(contour, point, False)

        # Return True if the point is inside or on the edge
        return result >= 0

    def select_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.line_points) < 2:
                self.line_points.append((x, y))
            if len(self.line_points) == 2:
                print(f"Selected Line: {self.line_points}")

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
        bowler_region = frame[y_min:y_max, x_min:x_max]

        # Detect shoes in the cropped region
        shoe_results = self.shoe_model(bowler_region, conf=0.6)
        
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
            if self.prev_bowler_bottom_right and self.prev_bowler_bottom_center and self.prev_bowler_bottom_left:
                _, self.prev_side = self.point_position(self.line_points, self.prev_bowler_bottom_right)
                _, self.curr_side = self.point_position(self.line_points, self.bowler_bottom_right)

                if self.prev_side == 'right' and self.curr_side == 'left' and not self.bowler_crossed_line:
                    self.bowler_crossed_line = True
                    self.put_text_on_frame(frame, '--Bowler crossed line--')
                    print("-------------------------------------------Bowler crossed line-------------------------------------------")
                    cv2.imwrite('/home/soumyadeep@quidich.local/soumyadeep/No_Ball/bowled.jpg', frame)

                if self.prev_bowler_bottom_center[0] < bottom_center[0] and self.prev_bowler_bottom_left[0] < self.bowler_bottom_left[0] and self.prev_bowler_bottom_right[0] < self.bowler_bottom_right[0]:
                    self.bowler_returning = True
                    self.put_text_on_frame(frame, f'--Bowler returning-- {self.prev_bowler_bottom_center[0], bottom_center[0], self.prev_bowler_bottom_left[0], self.bowler_bottom_left[0], self.prev_bowler_bottom_right[0], self.bowler_bottom_right[0]}')
                    print("-------------------------------------------Bowler returning-------------------------------------------")
                    cv2.imwrite('/home/soumyadeep@quidich.local/soumyadeep/No_Ball/bowler_returning.jpg', frame)
                    
                elif self.prev_bowler_bottom_center[0] > bottom_center[0] and self.prev_bowler_bottom_left[0] > self.bowler_bottom_left[0] and self.prev_bowler_bottom_right[0] > self.bowler_bottom_right[0]:
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
                if distance_to_line < 50:
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

    def draw_segmentation(self, shoe_seg, prompted_bbox, frame):
        masks = shoe_seg[0].masks.data.cpu().numpy()

        print(masks)

        # Create an empty image for the mask
        mask_image = np.zeros_like(frame)

        for mask in masks:
            mask  = mask.astype(np.uint8)

            resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            print(f"Resized mask shape: {resized_mask.shape}, Frame shape: {frame.shape}")
            bbox_mask = np.zeros_like(resized_mask)

            bbox_mask[prompted_bbox[1]:prompted_bbox[3], prompted_bbox[0]:prompted_bbox[2]] = resized_mask[prompted_bbox[1]:prompted_bbox[3], prompted_bbox[0]:prompted_bbox[2]]
            print(f"Non-zero values in bbox_mask: {np.count_nonzero(bbox_mask)}")
            mask_image[bbox_mask == 1] = [0, 255, 0]  # Color the mask (green in this case)
            
        segmented_image = cv2.addWeighted(frame, 0.7, mask_image, 0.3, 0)

        return segmented_image

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break

            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time)
            self.prev_time = curr_time

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect bowler
            bowler_results = self.bowler_model(frame, verbose=False, show=False)
            bowler_box = []
            fielder_box = []
            bowler_detected = False
            segmented_image = None

            for result in bowler_results:
                for boxes in result.boxes:
                    if boxes.cls == 1 and boxes.conf >= 0.5:
                        bowler_detected = True
                        x_min, y_min, x_max, y_max = map(int, boxes.xyxy[0])
                        bowler_box = [x_min, y_min, x_max, y_max]
                        if self.prev_fielder_box is None:
                            self.prev_fielder_box = bowler_box
                        frame = self.mark_bowler(frame, bowler_box)

                    elif boxes.cls == 4 and boxes.conf >= 0.4:
                        x_min, y_min, x_max, y_max = map(int, boxes.xyxy[0])
                        fielder_box = [x_min, y_min, x_max, y_max]
                        if self.prev_fielder_box is None:
                            self.prev_fielder_box = fielder_box

                        if bowler_box and fielder_box and len(bowler_box) == 4 and len(self.prev_fielder_box) == 4:
                            bowler_iou = self.calculate_iou(bowler_box, self.prev_fielder_box)

                            if bowler_iou >= 0.7:
                                self.prev_fielder_box = fielder_box
                                bowler_detected = True
                                frame = self.mark_bowler(frame, fielder_box)

                    elif boxes.cls == 10 and boxes.conf >= 0.2:
                        stump_x_min, stump_y_min, stump_x_max, stump_y_max = map(int, boxes.xyxy[0])
                        cv2.rectangle(frame, (stump_x_min, stump_y_min), (stump_x_max, stump_y_max), (0, 255, 0), 2)
                        self.stump_bottom_left = (stump_x_min, stump_y_max)

                    if bowler_detected and not self.bowler_returning:
                        self.persistent_centroid = self.find_persistent_max_y(self.min_centroids, self.frame_num)
                        if self.persistent_centroid and self.shoe_detected:
                            self.persistent_counter += 1
                            cv2.circle(frame, (int(self.persistent_centroid[0]), int(self.persistent_centroid[1])), 10, (255, 0, 0), -1)
                            cv2.putText(frame, f'Persistent Max Y', (int(self.persistent_centroid[0]), int(self.persistent_centroid[1]) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            if self.persistent_counter > fps:
                                self.persistent_counter = 0
                            seg_results = self.seg_model(source=frame.copy(), points=[self.persistent_centroid], conf=0.7)
                            segmented_image = self.draw_segmentation(seg_results, self.prompted_bbox, frame)

            if len(self.line_points) == 2:
                _, self.polygon_pts = self.draw_parallel_lines_and_roi(frame, self.line_points, self.stump_bottom_left)
                cv2.line(frame, self.line_points[0], self.line_points[1], (0, 255, 255), 5)

            cv2.imshow('Video', frame)

            if segmented_image is not None:
                cv2.imshow('Video', segmented_image)
                cv2.imwrite(f'./misc/foot_seg_frame{self.frame_num}.jpg', segmented_image)

            self.frame_num += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()





if __name__ == "__main__":

    detector = NoBallDetector(
    # bowler_model_path="/home/soumyadeep@quidich.local/soumyadeep/No_Ball/models/v11s-640-scrt.pt",
    bowler_model_path="./models/v11s-640-scrt.pt",
    shoe_model_path="./models/shoe_det_best_v1.pt",
    seg_model_path="./models/sam2_l.pt",
    video_path="/home/soumyadeep@quidich.local/soumyadeep/No_Ball/SHGN1_S001_S002_T238_deinterlaced.mp4"
    )
    detector.run()