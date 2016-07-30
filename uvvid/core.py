import cv2 as cv
import numpy as np


class UVVID:

    def __init__(self):
        self.prev_cursor_position = (None, None)

    def get_cursor_mask(self, hsv_frame):
        # Clear everything that is not white from the gray image. This should
        # leave us with just the cursor
        white_lower = np.array([0, 0, 0])
        white_upper = np.array([255, 25, 255])
        return cv.inRange(hsv_frame, white_lower, white_upper)

    def find_cursor(self, frame, template_frame, debug=True):
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = self.get_cursor_mask(hsv_frame)

        # The V part of HSV is actually the grayscale we are looking for.
        gray_frame = cv.split(hsv_frame)[2]
        gray_frame = cv.bitwise_and(gray_frame, gray_frame, mask=mask)

        res = cv.matchTemplate(gray_frame, template_frame, cv.TM_CCORR)

        min, max, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = max_loc

        bottom_right = (top_left[0] + template_frame.shape[0],
                        top_left[1] + template_frame.shape[1])

        cursor_center = (top_left[0] + template_frame.shape[0]//2,
                         top_left[1] + template_frame.shape[1]//2)

        if debug:
            cv.rectangle(frame, top_left, bottom_right,
                         (255, 255, 255), 1)

        return cursor_center

    def is_cursor_drawing(self, frame, prev_frame, cursor_coord, window=10,
                          min_drawing_pixels=0.2, debug=True):
        x, y = cursor_coord
        x_min = np.clip(x-window, 0, frame.shape[0])
        x_max = np.clip(x+window, 0, frame.shape[0])
        y_min = np.clip(y-window, 0, frame.shape[1])
        y_max = np.clip(y+window, 0, frame.shape[1])
        selection = frame[y_min:y_max, x_min:x_max]
        prev_selection = prev_frame[y_min:y_max, x_min:x_max]

        hsv_selection = cv.cvtColor(selection, cv.COLOR_BGR2HSV)
        hsv_prev_selection = cv.cvtColor(prev_selection, cv.COLOR_BGR2HSV)
        # Temporary fix for error coused by x_min and x_max having same value
        # so shape is (X, 0, X) which couses cvtColor return None
        if hsv_selection is None or hsv_prev_selection is None:
            return None, None

        black = np.zeros((selection.shape[0], selection.shape[1], 3), np.uint8)

        # Invert masks to filter out everything that is either the cursor or
        # the blackboard
        mask = 255 - self.get_cursor_mask(hsv_selection)
        prev_mask = 255 - self.get_cursor_mask(hsv_prev_selection)
        selection = cv.bitwise_or(selection, black, mask=mask)
        prev_selection = cv.bitwise_or(prev_selection, black, mask=prev_mask)
        diff = cv.absdiff(selection, prev_selection)

        kernel = np.ones((3, 3), np.uint8)
        diff = cv.erode(diff, kernel, iterations=1)
        diff = cv.dilate(diff, kernel, iterations=1)

        reshaped = diff[:, :, 2].reshape(-1)

        # Threshold residue from all the masking and eroding
        reshaped = np.reshape(cv.threshold(reshaped, 5, 255,
                                           cv.THRESH_BINARY)[1], (-1))
        nonzero = np.count_nonzero(reshaped)
        drawing = nonzero >= (window+1)**2 * min_drawing_pixels

        color_val = np.max(diff.reshape(-1, 3), axis=0).reshape((1, 1, 3))
        color = tuple(color_val[0][0].astype('int'))

        if debug:
            ds = "Drawing" if drawing else "Idle"
            cv.rectangle(frame, (frame.shape[1] - 60, 20),
                                (frame.shape[1] - 40, 40), color, -1)
            cv.putText(frame, ds, (frame.shape[1] - 150, 40),
                       cv.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 1)

        return drawing, color

    def find_cursor_color(self, frame, cursor_coord, window=10):
        pass
