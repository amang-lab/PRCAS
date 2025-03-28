"""
finding lane function reference:
https://github.com/mohamedameen93/Lane-lines-detection-using-Python-and-OpenCV
"""

import numpy as np
import cv2
import os
# from moviepy.editor import VideoFileClip

"""
This is the file for canny edge detection
"""
def RGB_color_selection(image):
    """
    Apply color selection to RGB images to blackout everything except for white and yellow lane lines.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    # White color mask
    lower_threshold = np.uint8([200, 200, 200])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower_threshold, upper_threshold)

    # Yellow color mask
    lower_threshold = np.uint8([175, 175, 0])
    upper_threshold = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower_threshold, upper_threshold)

    # Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image
def convert_hsv(image):
    """
    Convert RGB images to HSV.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def convert_hsl(image):
    """
    Convert RGB images to HSL.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
def HSL_color_selection(image):
    """
    Apply color selection to the HSL images to blackout everything except for white and yellow lane lines.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    # Convert the input image to HSL
    converted_image = convert_hsl(image)

    # White color mask
    lower_threshold = np.array([np.round(0 / 2), np.round(0.75 * 255), np.round(0.00 * 255)])
    upper_threshold = np.array([np.round(360 / 2), np.round(1.00 * 255), np.round(0.30 * 255)])
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    # Yellow color mask
    lower_threshold = np.array([np.round(40 / 2), np.round(0.00 * 255), np.round(0.35 * 255)])
    upper_threshold = np.array([np.round(60 / 2), np.round(1.00 * 255), np.round(1.00 * 255)])
    yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    # Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image
def gray_scale(image):
    """
    Convert images to gray scale.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def gaussian_smoothing(image, kernel_size = 13):
    """
    Apply Gaussian filter to the input image.
        Parameters:
            image: An np.array compatible with plt.imshow.
            kernel_size (Default = 13): The size of the Gaussian kernel will affect the performance of the detector.
            It must be an odd number (3, 5, 7, ...).
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def canny_detector(image, low_threshold = 50, high_threshold = 150):
    """
    Apply Canny Edge Detection algorithm to the input image.
        Parameters:
            image: An np.array compatible with plt.imshow.
            low_threshold (Default = 50).
            high_threshold (Default = 150).
    """
    return cv2.Canny(image, low_threshold, high_threshold)

def region_selection(image):
    """
    Determine and cut the region of interest in the input image.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    mask = np.zeros_like(image)
    #Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #We could have used fixed numbers as the vertices of the polygon,
    #but they will not be applicable to images with different dimesnions.
    rows, cols = image.shape[:2]
    bottom_left  = [cols * 0.1, rows * 0.8]
    top_left     = [cols * 0.3, rows * 0.6]
    bottom_right = [cols * 0.4, rows * 0.8]
    top_right    = [cols * 0.4, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    bottom_left = [cols * 0.6, rows * 0.8]
    top_left = [cols * 0.6, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.8]
    top_right = [cols * 0.7, rows * 0.6]

    vertices2 = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    combined_roi = np.concatenate((vertices, vertices2))

    cv2.fillPoly(mask, combined_roi, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def hough_transform(image):
    """
    Determine and cut the region of interest in the input image.
        Parameters:
            image: The output of a Canny transform.
    """
    rho = 1              #Distance resolution of the accumulator in pixels.
    theta = np.pi/180    #Angle resolution of the accumulator in radians.
    threshold = 20       #Only lines that are greater than threshold will be returned.
    minLineLength = 20   #Line segments shorter than that are rejected.
    maxLineGap = 300     #Maximum allowed gap between points on the same line to link them
    return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold,
                           minLineLength = minLineLength, maxLineGap = maxLineGap)

def draw_lines(image, lines, color = [255, 0, 0], thickness = 2):
    """
    Draw lines onto the input image.
        Parameters:
            image: An np.array compatible with plt.imshow.
            lines: The lines we want to draw.
            color (Default = red): Line color.
            thickness (Default = 2): Line thickness.
    """
    image = np.copy(image)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


def average_slope_intercept(lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
        Parameters:
            lines: The output lines from Hough Transform.
    """
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)
    if(lines is None):
        return (0,0),(0,0)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if(slope>=-1 and slope<=1): continue
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane
def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    if slope==0:
        slope=1
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    """
    Create full lenght lines from pixel points.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    # determine left is left right is right
    middlepoint=1280/2
    if(left_line is not None):
        if(left_line[0][0]>middlepoint or left_line[1][0]>middlepoint):
            left_line=None
    if(right_line is not None):
        if (right_line[0][0] < middlepoint or right_line[1][0] < middlepoint):
            right_line = None
    return left_line, right_line


def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    """
    Draw lines onto the input image.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
            color (Default = red): Line color.
            thickness (Default = 12): Line thickness.
    """
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def frame_processor(image):
    """
    Process the input frame to detect lane lines.
        Parameters:
            image: Single video frame.
    """
    color_select = HSL_color_selection(image)
    #color_select=RGB_color_selection(image)
    gray = gray_scale(color_select)
    smooth = gaussian_smoothing(gray)
    edges = canny_detector(smooth)
    region = region_selection(edges)
    hough = hough_transform(region)
    result = draw_lane_lines(image, lane_lines(image, hough))
    return result

def getlaneline(image):
    color_select = HSL_color_selection(image)
    gray = gray_scale(color_select)
    smooth = gaussian_smoothing(gray)
    #cv2.imshow("1.3", smooth)
    edges = canny_detector(smooth)
    #cv2.imshow("1.4", edges)
    region = region_selection(edges)
    #cv2.imshow("1", region)
    hough = hough_transform(region)
    # only get the lines
    return lane_lines(image, hough)
# def process_video(test_video, output_video):
#     """
#     Read input video stream and produce a video file with detected lane lines.
#         Parameters:
#             test_video: Input video.
#             output_video: A video file with detected lane lines.
#     """
#     # use RGB
#     input_video = VideoFileClip(os.path.join('test_videos', test_video), audio=False)
#     processed = input_video.fl_image(frame_processor)
#     processed.write_videofile(os.path.join('output_videos', output_video), audio=False)

def remove_bright_glare(img):
    hh, ww = img.shape[:2]
    # threshold
    lower = (150, 150, 150)
    upper = (240, 240, 240)
    thresh = cv2.inRange(img, lower, upper)

    # apply morphology close and open to make mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel, iterations=1)

    # floodfill the outside with black
    black = np.zeros([hh + 2, ww + 2], np.uint8)
    mask = morph.copy()
    mask = cv2.floodFill(mask, black, (0, 0), 0, 0, 0, flags=8)[1]

    # use mask with input to do inpainting
    #super slow
    result1 = cv2.inpaint(img, mask, 101, cv2.INPAINT_TELEA)
    result2 = cv2.inpaint(img, mask, 101, cv2.INPAINT_NS)
    cv2.imshow("RESULT1", result1)
    cv2.imshow("RESULT2", result2)

def test():
    VIDEO_Path = './test/Camera_xhs_1706431332907.mp4'
    # VIDEOUT_Path = 'C:/Users/clari/Desktop/Graduate project/App/test/車CAM直擊 - 8號仔！邊個跌咁大袋野係路中心呀....花L晒喇！！ #希望有人影到2.mp4'
    #process_video(VIDEO_Path, VIDEOUT_Path)
    #user bgr
    cap = cv2.VideoCapture(VIDEO_Path)
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("YOLOv8 Tracking", rgb_frame)

            annotated_frame = frame_processor(rgb_frame)
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # TODO: tune the image parameter
    # TODO: try to find a stable line the is suitable then won't process the frame again

#test()