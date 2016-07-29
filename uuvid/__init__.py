import numpy as np
import cv2 as cv


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
            cv.rectangle(gray_frame, top_left, bottom_right,
                         (255, 255, 255), 1)
            cv.imshow('gray', gray_frame)

        return cursor_center

    def is_cursor_drawing(self, frame, prev_frame, cursor_coord, window=10,
                          min_drawing_pixels=0.3, debug=True):
        x, y = cursor_coord
        x_min = np.clip(x-window, 0, frame.shape[0])
        x_max = np.clip(x+window, 0, frame.shape[0])
        y_min = np.clip(y-window, 0, frame.shape[1])
        y_max = np.clip(y+window, 0, frame.shape[1])

        selection = frame[y_min:y_max, x_min:x_max]
        prev_selection = prev_frame[y_min:y_max, x_min:x_max]

        hsv_selection = cv.cvtColor(selection, cv.COLOR_BGR2HSV)
        hsv_prev_selection = cv.cvtColor(prev_selection, cv.COLOR_BGR2HSV)

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
        if debug:
            cv.imshow('diff', diff)

        reshaped = diff.reshape(-1, 3)
        nonzero = np.count_nonzero(reshaped)//3
        color = np.max(reshaped, axis=0)

        drawing = nonzero >= (window+1)**2 * min_drawing_pixels
        return drawing, color

    def find_cursor_color(self, frame, cursor_coord, window=10):
        pass

if __name__ == "__main__":
    uvvid = UVVID()
    template_frame = cv.imread('cursor.png', 0)
    cap = cv.VideoCapture('exponents.mp4')
    prev_frame = None
    while cap.isOpened():
        ret, frame = cap.read()

        coord = uvvid.find_cursor(frame, template_frame)
        if prev_frame is not None:
            print uvvid.is_cursor_drawing(frame, prev_frame, coord)
        cv.imshow('frame', frame)
        prev_frame = frame
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
