import numpy as np
import cv2 as cv


class UVVID:

    def __init__(self):
        self.prev_cursor_position = (None, None)

    def find_cursor(self, frame, template_frame, debug=True):
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Clear everything that is not white from the gray image. This should
        # leave us with just the cursor
        white_lower = np.array([0, 0, 0])
        white_upper = np.array([255, 25, 255])

        mask = cv.inRange(hsv_frame, white_lower, white_upper)

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
            cv.rectangle(frame, top_left, bottom_right,
                         (255, 255, 255), 1)

        return cursor_center

if __name__ == "__main__":
    uvvid = UVVID()
    template_frame = cv.imread('cursor.png', 0)
    cap = cv.VideoCapture('exponents.mp4')
    while cap.isOpened():
        ret, frame = cap.read()

        uvvid.find_cursor(frame, template_frame)
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
