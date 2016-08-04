import math

import cv2 as cv

import numpy as np


class UVVID:

    def __init__(self):
        self.strokes = []
        self.idle_cursors = []
        self.cursor_points = []
        self.colors = []
        self.frame_time_diff = 1
        self.before_frame = None
        self.old_frame = None
        self.drawing_points = []

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

    def remove_cursor(self, frame1, frame2=None, ker=(3, 3)):
        # Invert masks to filter out everything that is either the cursor or
        # the blackboard
        black = np.zeros((frame1.shape[0], frame1.shape[1], 3), np.uint8)
        mask = 255 - self.get_cursor_mask(frame1)
        selection = cv.bitwise_or(frame1, black, mask=mask)
        if frame2 is not None:
            prev_mask = 255 - self.get_cursor_mask(frame2)
            prev_selection = cv.bitwise_or(frame2, black, mask=prev_mask)
            diff = cv.absdiff(selection, prev_selection)
        else:
            diff = selection
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

    def __add_points(self, cursor_points, frame, before_frame):
        diff = cv.absdiff(frame, before_frame)
        diff = self.remove_cursor(diff)
        diff = cv.split(cv.cvtColor(diff, cv.COLOR_BGR2HSV))[2]
        _, diff = cv.threshold(diff, 50, 255, cv.THRESH_BINARY)
        cnts, _ = cv.findContours(diff, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:2]
        if len(cnts) <= 0:
            return False
        self.drawing_points.append(cursor_points)
        # add new shape to list of strokes
        self.strokes.append([])
        for cnt in cnts:
            point = cnt[0][0]
            self.strokes[-1].append((point[0], point[1]))
            for cnt_point in cnt:
                cx, cy = cnt_point[0]
                px, py = point
                if px-3 <= cx <= px+3 and py-3 <= cy <= py+3:
                    continue
                self.strokes[-1].append((cx, cy))
                point = cnt_point[0]
        return True

    def generate_strokes(self, frame, prev_frame, template_frame,
                         frame_diff=3, diff_degree=5):
        cursor = self.find_cursor(frame, template_frame)
        self.idle_cursors.append(cursor)
        if prev_frame is not None:
            is_drawing, color = self.is_cursor_drawing(frame,
                                                       prev_frame,
                                                       cursor)
        if is_drawing:
            if self.frame_time_diff > frame_diff:
                self.before_frame = self.old_frame
                # start with first cursor position before we started to draw
                pos = self.idle_cursors.pop()
                # append last 2 positions of the cursor
                self.cursor_points.append([self.idle_cursors.pop(-2), pos])
            self.frame_time_diff = 0
            self.cursor_points[-1].append(cursor)
        else:
            if self.frame_time_diff == 2 and self.before_frame is not None:
                found_conture = self.__add_points(self.cursor_points[-1],
                                                  frame,
                                                  self.before_frame)
                # add last cursor position after drawing
                if found_conture:
                    self.cursor_points[-1].append(self.idle_cursors.pop(-3))
            self.frame_time_diff += 1
        self.old_frame = prev_frame

    def get_strokes(self):
        return self.strokes

    def get_idle_cursor_positions(self):
        return self.idle_positions

    def __debug_points__(self, frame, points, show_cursor_points=False):
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
            cv.rectangle(frame, (min_x, min_y), (max_x, max_y),
                         (255, 255, 255), 1)
            for point in shape:
                cv.circle(frame, tuple(point), 2, (0, 0, 255), -1)
            if show_cursor_points:
                for cursor_point in self.drawing_points[i]:
                    cv.circle(frame, cursor_point, 3, (255, 0, 0), -1)

    def generate_json(self):
        pass
