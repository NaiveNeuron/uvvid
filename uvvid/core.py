import math

import cv2 as cv

import numpy as np


class UVVID:

    def __init__(self):
        self.prev_cursor_position = (None, None)
        self.strokes = []
        self.idle_positions = []
        self.colors = []
        self.frame_time_diff = 0
        self.nodraw_frame = None

    def get_cursor_mask(self, hsv_frame):
        # Clear everything that is not white from the gray image. This should
        # leave us with just the cursor
        white_lower = np.array([0, 0, 0])
        white_upper = np.array([255, 25, 255])
        return cv.inRange(hsv_frame, white_lower, white_upper)

    def find_cursor(self, frame, template_frame):
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = self.get_cursor_mask(hsv_frame)

        # The V part of HSV is actually the grayscale we are looking for.
        gray_frame = cv.split(hsv_frame)[2]
        gray_frame = cv.bitwise_and(gray_frame, gray_frame, mask=mask)

        res = cv.matchTemplate(gray_frame, template_frame, cv.TM_CCORR)
        min, max, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = max_loc

        cursor_center = (top_left[0] + template_frame.shape[0]//2,
                         top_left[1] + template_frame.shape[1]//2)
        self.cursor_coord = cursor_center

        return cursor_center

    def remove_cursor(self, frame1, frame2, ker=(3, 3)):
        # Invert masks to filter out everything that is either the cursor or
        # the blackboard
        black = np.zeros((frame1.shape[0], frame2.shape[1], 3), np.uint8)
        mask = 255 - self.get_cursor_mask(frame1)
        prev_mask = 255 - self.get_cursor_mask(frame2)
        selection = cv.bitwise_or(frame1, black, mask=mask)
        prev_selection = cv.bitwise_or(frame2, black, mask=prev_mask)
        diff = cv.absdiff(selection, prev_selection)
        kernel = np.ones(ker, np.uint8)
        diff = cv.erode(diff, kernel, iterations=1)
        diff = cv.dilate(diff, kernel, iterations=1)
        return diff

    def is_cursor_drawing(self, frame, prev_frame, cursor_coord, window=10,
                          min_drawing_pixels=0.2):
        x, y = cursor_coord
        # shape[1] is width of the frame
        x_min = np.clip(x-window, 0, frame.shape[1])
        x_max = np.clip(x+window, 0, frame.shape[1])
        # shape[0] is height of the frame
        y_min = np.clip(y-window, 0, frame.shape[0])
        y_max = np.clip(y+window, 0, frame.shape[0])

        selection = frame[y_min:y_max, x_min:x_max]
        prev_selection = prev_frame[y_min:y_max, x_min:x_max]

        hsv_selection = cv.cvtColor(selection, cv.COLOR_BGR2HSV)
        hsv_prev_selection = cv.cvtColor(prev_selection, cv.COLOR_BGR2HSV)

        diff = self.remove_cursor(hsv_selection, hsv_prev_selection)
        reshaped = diff[:, :, 2].reshape(-1)

        # Threshold residue from all the masking and eroding
        reshaped = np.reshape(cv.threshold(reshaped, 3, 255,
                                           cv.THRESH_BINARY)[1], (-1))
        nonzero = np.count_nonzero(reshaped)
        drawing = nonzero >= (window+1)**2 * min_drawing_pixels

        color_val = np.max(diff.reshape(-1, 3), axis=0).reshape((1, 1, 3))
        color = tuple(color_val[0][0].astype('int'))

        return drawing, color

    def __get_angle(self, new_point, last_point, in_degrees=True):
        new_x, new_y = new_point
        last_x, last_y = last_point
        theta = math.atan2(new_y - last_y, new_x - last_y)
        if in_degrees:
            return (math.degrees(theta) + 360) % 360
        return theta

    def __add_points(self, cursor, frame, template_frame):
        last_x, last_y = self.idle_positions[-1]
        x, y = cursor
        temp_width, temp_height = template_frame.shape
        x_max, x_min = max(last_x, x)+2, min(last_x, x)-2
        y_max, y_min = max(last_y, y)+2, min(last_y, y)-2
        selection = frame[y_min:y_max, x_min:x_max]
        nodraw_selection = self.nodraw_frame[y_min:y_max, x_min:x_max]
        selection = cv.cvtColor(selection, cv.COLOR_BGR2HSV)
        nodraw_selection = cv.cvtColor(nodraw_selection, cv.COLOR_BGR2HSV)
        diff = self.remove_cursor(selection, nodraw_selection, ker=(2, 2))
        diff = cv.split(diff)[2]
        _, reshaped = cv.threshold(diff, 3, 255, cv.THRESH_BINARY)
        points = np.transpose(np.nonzero(diff))
        for point in points:
            self.strokes[-1].append((x_min + point[1], y_min + point[0]))

    switch_flag = False

    def generate_strokes(self, frame, prev_frame, template_frame,
                         frame_diff=3, diff_degree=5):
        cursor = self.find_cursor(frame, template_frame)
        if prev_frame is not None:
            is_drawing, color = self.is_cursor_drawing(frame,
                                                       prev_frame,
                                                       cursor)
        if is_drawing:
            if self.frame_time_diff < frame_diff:
                if self.switch_flag:
                    self.__add_points(cursor, frame, template_frame)
                    self.switch_flag = False
                current_shape = self.strokes[-1]
                current_shape.append(cursor)
                self.nodraw_frame = frame
            else:
                self.colors.append(color)
                self.strokes.append([cursor])
                self.__add_points(cursor, frame, template_frame)
                self.switch_flag = True
            self.frame_time_diff = 0
        else:
            if self.frame_time_diff <= 1 and self.nodraw_frame is not None:
                self.__add_points(cursor, frame, template_frame)
            self.idle_positions.append(cursor)
            self.frame_time_diff += 1
            self.nodraw_frame = frame

    def get_strokes(self):
        return self.strokes

    def get_idle_cursor_positions(self):
        return self.idle_positions

    def __debug_points__(self, frame, points):
        '''
        Debug function to vizualize where are found
        cursor points located

        Args:
            frame - frame where to visualize the points (Warning: this will
                    modify passed frame)
            points - array of points to vizualize
        '''
        for i, shape in enumerate(points):
            arr_shape = np.asarray(shape).swapaxes(0, 1)
            min_x, min_y = np.min(arr_shape[0]), np.min(arr_shape[1])
            max_x, max_y = np.max(arr_shape[0]), np.max(arr_shape[1])
            cv.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 1)
            for point in shape:
                cv.circle(frame, tuple(point), 3, (0, 0, 255), -1)

    def generate_json(self):
        pass
