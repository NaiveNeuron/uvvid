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
        self.before_cursor = None
        self.old_frame = None
        self.threshold = 50

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

    def closest_node(self, point, nodes):
        if nodes.size == 0:
            return None, None
        deltas = nodes - point
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        return np.argmin(dist_2), dist_2

    def closest_nodes(self, points, nodes):
        points = np.asarray(points)
        deltas = nodes[:, np.newaxis, :] - points[:]
        dist_2 = np.einsum('xij,xij->xi', deltas, deltas)
        return np.argmin(dist_2, axis=0), dist_2

    def get_conture_points(self, frame, before_frame):
        diff = self.remove_cursor(frame, before_frame)
        diff = cv.split(cv.cvtColor(diff, cv.COLOR_BGR2HSV))[2]
        _, diff = cv.threshold(diff, 10, 255, cv.THRESH_BINARY)
        cnts, _ = cv.findContours(diff, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:2]
        if len(cnts) <= 0:
            return None
        return cnts

    def __insert_point__(self, arr, idx, point, frame):
        after_idx = idx+1 if idx+1 < len(arr) else idx
        before_idx = idx-1 if idx-1 >= 0 else idx
        after = arr[after_idx]
        before = arr[before_idx]
        _, dist1 = self.closest_nodes(np.asarray([point]),
                                      np.asarray([before, arr[idx], after]))
        _, dist2 = self.closest_nodes(np.asarray([arr[idx]]),
                                      np.asarray([before, point, after]))
        if dist2[1] <= self.threshold or min(dist1) >= 1000:
            return arr
        before_dist = dist1[0] + dist2[1] + dist2[2]
        after_dist = dist1[2] + dist2[1] + dist2[0]

        if idx == before_idx:
            before_dist -= dist1[0]
        elif idx == after_idx:
            after_dist -= dist1[2]

        if before_dist > after_dist:
            idx += 1
        return np.insert(arr, idx, point, axis=0)

    def __debug_sort__(self, points, frame):
        debug = frame.copy()
        for point in points:
            cv.circle(debug, tuple(point), 2, (255, 0, 255), -1)
            cv.imshow('debug sort', debug)
            cv.waitKey(0)

    def __add_points(self, cursor_points, cnts, frame):
        if len(cnts) == 1:
            conture_points = np.squeeze(cnts[0])
        else:
            conture_points = np.concatenate((np.squeeze(cnts[0]),
                                             np.squeeze(cnts[1])))
        if cv.contourArea(conture_points) <= 40:
            return

        num_points = int(conture_points.shape[0] / 10)
        num_points = num_points if num_points <= 5 else 5
        num_points = 2 if num_points <= 2 else num_points

        closest_idx, dist = self.closest_nodes(cursor_points, conture_points)
        dist_idx = np.argsort(np.swapaxes(dist, 0, 1), axis=1)[:, :num_points]
        closest_points = conture_points[closest_idx]
        conture_points = np.delete(conture_points, dist_idx.flatten(), axis=0)

        _, test_dist = self.closest_nodes(closest_points, closest_points)
        tmp = test_dist > 100
        cursor_dist = test_dist * tmp
        for i, c_dist in enumerate(cursor_dist):
            a = np.asarray(np.nonzero(c_dist == 0))
            remove = a[np.nonzero(a * (a != i))]
            if remove.size != 0:
                closest_points = np.delete(closest_points, remove, axis=0)

        for i, point in enumerate(conture_points):
            closest_idx, dist = self.closest_node(point, conture_points)
            dist_idx = np.argsort(dist)[:num_points]
            if max(dist) > self.threshold:
                clst_idx, _ = self.closest_node(point, closest_points)
                closest_points = self.__insert_point__(closest_points,
                                                       clst_idx, point, frame)
                continue
            mean = np.mean(conture_points[dist_idx], axis=0)
            mean = [int(mean[0]), int(mean[1])]
            _, mean_dist = self.closest_node(mean, np.asarray(closest_points))

            if mean_dist is None or min(mean_dist) > self.threshold:
                clst_idx, _ = self.closest_node(mean, closest_points)
                closest_points = self.__insert_point__(closest_points,
                                                       clst_idx, mean, frame)
        self.__debug_sort__(closest_points, frame)
        # add new shape to list of strokes
        self.strokes.append(closest_points)

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
                self.cursor_points.append([self.idle_cursors.pop()])
            self.frame_time_diff = 0
            self.cursor_points[-1].append(cursor)
        else:
            if self.frame_time_diff == 2 and self.before_frame is not None:
                found_conture = self.get_conture_points(frame,
                                                        self.before_frame)
                # add last cursor position after drawing
                if found_conture is not None:
                    self.cursor_points[-1].append(self.idle_cursors.pop(-3))
                    self.__add_points(self.cursor_points[-1], found_conture,
                                      frame)

            self.frame_time_diff += 1
        self.old_cursor = cursor
        self.old_frame = prev_frame

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
            arr_shape = np.asarray(shape)
            if arr_shape.size == 0:
                continue
            arr_shape = arr_shape.swapaxes(0, 1)
            min_x, min_y = np.min(arr_shape[0]), np.min(arr_shape[1])
            max_x, max_y = np.max(arr_shape[0]), np.max(arr_shape[1])
            cv.rectangle(frame, (min_x, min_y), (max_x, max_y),
                         (255, 255, 255), 1)
            for point in shape:
                cv.circle(frame, tuple(point), 2, (0, 0, 255), -1)

    def generate_json(self):
        pass
