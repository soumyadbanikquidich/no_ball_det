import torch
from ultralytics import YOLO, SAM
import cv2
import numpy as np
import time
# import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from shapely.geometry import Point, Polygon

line_points = []
max_y_persistent_peak = None
max_y_persistent_centroid = None
persistent_centroid = None
min_centroids = []
bowler_bottom_rights = []
max_peak_y = None
stump_bottom_left = None
polygon_pts = None
prev_centroid = None
persistent_counter = 0
bowler_bottom_right = None
prev_bowler_bottom_right = None
prev_bowler_bottom_left = None
prev_bowler_bottom_center = None
detect_persistency = False
prev_side = None
curr_side = None
bowler_crossed_line = False
bowler_bottom_left = None
bowler_bottom_points = None
bowler_returning = False
prompted_bbox = []
shoe_detected = False

def put_text_on_frame(text):

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

def calculate_iou(box1, box2):
    if len(box1) != 4 or len(box2) != 4:
        return 0

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)


    iou = interArea / float(box1Area + box2Area - interArea)

    return iou

# Assume we keep track of the previous bounding box
prev_shoe_box = None
prev_fielder_box = None

# Minimum IoU threshold to match the shoe in subsequent frames
iou_threshold = 0.3

def process_shoe_detections(shoe_results, x_min, y_min, frame):
    global prev_shoe_box, prev_centroid, prompted_bbox
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

            centroid = calculate_centroid([sx_min, sy_min, sx_max, sy_max])
            centroids.append(centroid)

            # Find the leftmost shoe
            if centroids:
                min_x_point = min(centroids, key=lambda c: c[0])
                if prev_centroid is None or prev_centroid[0] > min_x_point[0] and not bowler_returning:
                    print('prev_centroid: +++++++++++++++++++++++++++', prev_centroid)                    
                    print('min_x: +++++++++++++++++++++++++++++++++++++++++++++++++', min_x_point)
                    min_centroids.append(min_x_point)

                elif bowler_returning:
                    min_centroids.clear()

                prev_centroid = min_x_point
            
            prompted_bbox = [sx_min, sy_min, sx_max, sy_max]  # [x1, y1, x2, y2]


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
                if prev_shoe_box is None:
                    prev_shoe_box = current_shoe_box

                # Calculate IoU to match the current shoe with the previous one
                iou = calculate_iou(prev_shoe_box, current_shoe_box)

                if iou >= iou_threshold:
                    # Update the previous shoe bounding box
                    prev_shoe_box = current_shoe_box

                # Draw the bounding box and centroid for the leftmost shoe
                cv2.rectangle(frame, (sx_min, sy_min), (sx_max, sy_max), (255, 0, 0), 2)
                cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 5, (0, 0, 255), -1)
                new_cent = (centroid[0], centroid[1])

    return frame, min_centroids
    # return None

# Example usage:
# shoe_results = shoe_model(bowler_region, conf=0.8)
# frame = process_shoe_detections(shoe_results, x_min, y_min, frame)

def create_parallel_line_through_point(point, line_points):
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

def draw_parallel_lines_and_roi(image, line_points, point):
    """
    Draws the original line, the parallel line through the given point, and the ROI joining the two lines.
    
    :param image: The image on which to draw.
    :param line_points: List of two tuples [(x1, y1), (x2, y2)] representing the original line.
    :param point: Tuple of (x, y) representing the point through which the parallel line should pass.
    :return: Image with the drawn lines and ROI.
    """
    # Create the parallel line through the given point
    parallel_line = create_parallel_line_through_point(point, line_points)

    # Draw the original line
    cv2.line(image, line_points[0], line_points[1], (0, 255, 0), 2)

    # Draw the parallel line
    cv2.line(image, parallel_line[0], parallel_line[1], (0, 255, 0), 2)

    # Create the ROI by connecting the lines into a quadrilateral
    pts = np.array([line_points[0], line_points[1], parallel_line[1], parallel_line[0]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    return image, pts

def is_point_in_polygon(point, polygon_points):
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

def select_points(event, x, y, flags, param):
    global line_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_points) < 2:
            line_points.append((x, y))
        if len(line_points) == 2:
            print(f"Selected Line: {line_points}")

def point_position(line, P):
    A, B = line_points
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

def point_line_distance(point, line):
    x0, y0 = point
    (x1, y1), (x2, y2) = line
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return numerator / denominator if denominator != 0 else float('inf')

def detect_peaks(y_values, persistence_threshold=3):
    # y_values = list(map(lambda x:x[1], centroids))
    peaks, properties = find_peaks(y_values, prominence=1, distance=persistence_threshold)
    return peaks

# def get_persistent_peak(centroids, frame_num):
#     global max_y_persistent_peak

#     centroid_x_values = list(map(lambda x:x[0], centroids))
#     centroid_y_values = list(map(lambda x:x[1], centroids))

#     peaks = detect_peaks(centroid_y_values)
#     if centroids and len(line_points) == 2:
#         if peaks.size > 0:
#             max_peak_index = peaks[-1]
#             if max_peak_y>centroid_y_values[max_peak_index]:
#                 max_peak = (centroid_x_values[max_peak_index], centroid_y_values[max_peak_index])
#             else:
#                 max_peak_y = centroid_y_values[max_peak_index]:
#             distance_to_line = point_line_distance(max_peak, line_points)
#             if distance_to_line < 50:
#                 print("\n=======================================================================================================\n")
#                 print(max_peak, max_peak_index)
#                 if max_y_persistent_peak is None or frame_num - max_y_persistent_peak['frame_num'] >= 3:
#                     max_y_persistent_peak = {'peak': max_peak, 'frame_num': frame_num}
#                     return max_peak
#     return None

def get_persistent_peak(centroids, frame_num):
    global max_y_persistent_peak
    global max_peak_y

    # Initialize max_peak_y if it hasn't been initialized before
    if max_peak_y is None:
        max_peak_y = float('-inf')

    centroid_x_values = list(map(lambda x: x[0], centroids))
    centroid_y_values = list(map(lambda x: x[1], centroids))

    # Detect peaks in the y-values
    peaks = detect_peaks(centroid_y_values)

    if centroids and len(line_points) == 2:
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
            distance_to_line = point_line_distance(max_peak, line_points)

            # Check if the peak is within the desired range from the line
            if distance_to_line < 50:
                print("\n=======================================================================================================\n")
                print(max_peak, max_peak_index)

                # Update max_y_persistent_peak if it's the first peak or if enough frames have passed
                if max_y_persistent_peak is None or frame_num - max_y_persistent_peak['frame_num'] >= 3:
                    max_y_persistent_peak = {'peak': max_peak, 'frame_num': frame_num}
                    return max_peak

    return None

def mark_bowler(frame, bbox):
    global bowler_bottom_right, prev_bowler_bottom_right, prev_bowler_bottom_left, detect_persistency, bowler_crossed_line, bowler_bottom_left, bowler_bottom_points, bowler_returning, prev_bowler_bottom_center, shoe_detected

    x_min, y_min, x_max, y_max = bbox

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Crop bowler region
    bowler_region = frame[y_min:y_max, x_min:x_max]

    # Detect shoes in the cropped region
    shoe_results = shoe_model(bowler_region, conf=0.6)
    
    if len(shoe_results) != 0:
        shoe_detected = True
    else:
        shoe_detected = False
    centroids = []
    centroid_x = (x_min + x_max) / 2
    bottom_center = (centroid_x, y_max)
    bowler_bottom_right = (x_max, y_max)
    bowler_bottom_left = (x_min, y_max)

    bowler_bottom_points = [(int(x), y_max) for x in np.linspace(x_min, x_max, 5)]

    # cross_product, pos = point_position(line_points, bowler_bottom_right)

    if bowler_bottom_right and bottom_center and bowler_bottom_left:
        if prev_bowler_bottom_right and prev_bowler_bottom_center and prev_bowler_bottom_left:
            _, prev_side = point_position(line_points, prev_bowler_bottom_right)
            _, curr_side = point_position(line_points, bowler_bottom_right)

            if prev_side == 'right' and curr_side == 'left' and not bowler_crossed_line:
                bowler_crossed_line = True
                put_text_on_frame('--Bowler crossed line--')
                print("-------------------------------------------Bowler crossed line-------------------------------------------")
                cv2.imwrite('/home/soumyadeep@quidich.local/soumyadeep/No_Ball/bowled.jpg', frame)

            if prev_bowler_bottom_center[0] < bottom_center[0] and prev_bowler_bottom_left[0] < bowler_bottom_left[0] and prev_bowler_bottom_right[0] < bowler_bottom_right[0]:
                bowler_returning = True
                put_text_on_frame(f'--Bowler returning-- {prev_bowler_bottom_center[0], bottom_center[0], prev_bowler_bottom_left[0], bowler_bottom_left[0], prev_bowler_bottom_right[0], bowler_bottom_right[0]}')
                print("-------------------------------------------Bowler returning-------------------------------------------")
                cv2.imwrite('/home/soumyadeep@quidich.local/soumyadeep/No_Ball/bowler_returning.jpg', frame)
                
            elif prev_bowler_bottom_center[0] > bottom_center[0] and prev_bowler_bottom_left[0] > bowler_bottom_left[0] and prev_bowler_bottom_right[0] > bowler_bottom_right[0]:
                bowler_returning = False


        prev_bowler_bottom_right = bowler_bottom_right
        prev_bowler_bottom_center = bottom_center
        prev_bowler_bottom_left = bowler_bottom_left

    if is_point_in_polygon(bottom_center, polygon_pts):
        cv2.circle(frame, (int(centroid_x), int(y_max)), 4, (0, 0, 255), -1)
        frame, min_centroids = process_shoe_detections(shoe_results, x_min, y_min, frame)

    return frame

def find_persistent_max_y(centroids, frame_num, persistence_threshold=3):
    global max_y_persistent_centroid
    check_for_foot = False
    
    if centroids and len(line_points) == 2:
        # sorted_centroids = sorted(centroids, key=lambda c: c[1], reverse=True)
        sorted_centroids = sorted(centroids, key=lambda c: (-c[1], c[0]))  #sort with max_y them min_x
        for centroid in sorted_centroids:
            distance_to_line = point_line_distance(centroid, line_points)
            if distance_to_line < 50:
                check_for_foot = True
            elif is_point_in_polygon(centroid, polygon_pts):
                check_for_foot = True
            else:
                check_for_foot = False 
                
            if check_for_foot:
                if max_y_persistent_centroid is None or frame_num - max_y_persistent_centroid['frame_num'] >= persistence_threshold:
                    max_y_persistent_centroid = {'centroid': centroid, 'frame_num': frame_num}
                    return centroid
    return None

def calculate_centroid(bbox):
    x_min, y_min, x_max, y_max = bbox
    centroid_x = (x_min + x_max) / 2
    centroid_y = (y_min + y_max) / 2
    return [centroid_x, centroid_y]

def draw_segmentation(shoe_seg, prompted_bbox, frame):
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


# bowler_model = YOLO('/home/soumyadeep@quidich.local/soumyadeep/No_Ball/v11s3-1088-scrt.pt')
bowler_model = YOLO("/home/soumyadeep@quidich.local/soumyadeep/No_Ball/models/v11s-640-scrt.pt")
shoe_model = YOLO('/home/soumyadeep@quidich.local/soumyadeep/No_Ball/runs/detect/train5/weights/shoe_det_best_v1.pt')
seg_model = SAM('./models/sam2_l.pt')

bowler_model.to(0)
shoe_model.to(0)
seg_model.to(0)


video_path = "/home/soumyadeep@quidich.local/soumyadeep/No_Ball/SHGN1_S001_S002_T238_deinterlaced.mp4"
# video_path = "/home/quidich/z-cam/Untitled_mark_T09-28-51-247_cam_6.mp4"
# video_path = "/home/soumyadeep@quidich.local/soumyadeep/No_Ball/z-cam/Untitled_mark_T09-35-15-631_cam_3.mp4"
# video_path = "/home/soumyadeep@quidich.local/soumyadeep/No_Ball/IND_BAN_TEST_1.MOV"

cap = cv2.VideoCapture(video_path)

video_name = video_path.split('/')[-1].split('.')[0]



if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

centroid_y_values = []
frame_num = 0
prev_time = time.time()


cv2.namedWindow('Video')
cv2.setMouseCallback('Video', select_points)


while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect bowler
    bowler_results = bowler_model(frame, verbose=False, show=False)
    bowler_box = []
    fielder_box = []
    bowler_detected = False
    # shoe_detected = False
    
    segmented_image = None
    
    for result in bowler_results:
        for boxes in result.boxes:

            if boxes.cls == 1 and boxes.conf >= 0.5:
                # print(boxes)
                bowler_detected = True
                x_min, y_min, x_max, y_max = map(int, boxes.xyxy[0])
                bowler_box = [x_min, y_min, x_max, y_max]
                if prev_fielder_box is None:
                    prev_fielder_box = bowler_box
                frame = mark_bowler(frame, bowler_box)

            elif boxes.cls == 4 and boxes.conf >= 0.4:
                # print(boxes)
                x_min, y_min, x_max, y_max = map(int, boxes.xyxy[0])
                fielder_box = [x_min, y_min, x_max, y_max]
                if prev_fielder_box is None:
                    prev_fielder_box = fielder_box

                if bowler_box and fielder_box and len(bowler_box) == 4 and len(prev_fielder_box) == 4:
                    # fielder_box.append(boxes)
                    bowler_iou = calculate_iou(bowler_box, prev_fielder_box)

                    if bowler_iou >= 0.7:
                        prev_fielder_box = fielder_box
                        print(bowler_iou)
                        bowler_detected = True
                        frame = mark_bowler(frame, fielder_box)
            
            elif boxes.cls == 10 and boxes.conf >= 0.2:
                # print(boxes)
                stump_x_min, stump_y_min, stump_x_max, stump_y_max = map(int, boxes.xyxy[0])
                # bowler_box.append(boxes)

                cv2.rectangle(frame, (stump_x_min, stump_y_min), (stump_x_max, stump_y_max), (0, 255, 0), 2)

                stump_bottom_left = (stump_x_min, stump_y_max)


            if bowler_detected and not bowler_returning:
            #     for bowler_bottom_point in bowler_bottom_points:
            #         if is_point_in_polygon(bowler_bottom_point, polygon_pts):
                persistent_centroid = find_persistent_max_y(min_centroids, frame_num)
                print('min_centroids: ==================', min_centroids)
                print('persistent_max: ==================', persistent_centroid)

                        # persistent_centroid = get_persistent_peak(min_centroids, frame_num)

                if persistent_centroid and shoe_detected:
                    persistent_counter+=1
                    cv2.circle(frame, (int(persistent_centroid[0]), int(persistent_centroid[1])), 10, (255, 0, 0), -1)
                    cv2.putText(frame, f'Persistent Max Y', (int(persistent_centroid[0]), int(persistent_centroid[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    # cv2.imwrite(f'/home/soumyadeep@quidich.local/soumyadeep/No_Ball/peak_point_frames/{video_name}_max_y_frame_{frame_num}.jpg', frame)
                    if persistent_counter>fps:
                    # if bowler_returning:
                        # persistent_centroid = None
                        # min_centroids.clear()
                        persistent_counter = 0
                        
                    seg_results = seg_model(source=frame.copy(), points=[persistent_centroid], conf=0.7)
                    segmented_image = draw_segmentation(seg_results, prompted_bbox, frame)
                    # frame = segmented_image

    # elif not bowler_detected:
    #     min_centroids.clear()

            # if centroid_y_values:
            #     plot_shoe_trajectory(centroid_y_values, frame_num)
    if len(line_points) == 2:
        _, polygon_pts = draw_parallel_lines_and_roi(frame, line_points, stump_bottom_left)
        cv2.line(frame, line_points[0], line_points[1], (0, 255, 255), 5)

    cv2.imshow('Video', frame)
    
    if segmented_image is not None:
        cv2.imshow('Video', segmented_image)
        cv2.imwrite(f'foot_seg_frame{frame_num}.jpg', segmented_image)

    frame_num += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()