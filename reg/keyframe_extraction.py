import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import math
from sklearn.cluster import KMeans

class OpticalFlowCalculator:
    '''
    A class for optical flow calculations using OpenCV
    '''
    def __init__(self,
                 frame_width,
                 frame_height,
                 scaledown=1,
                 perspective_angle=0,
                 move_step=16,
                 window_name=None,
                 flow_color_rgb=(0, 255, 0)):
        '''
        Creates an OpticalFlow object for images with specified width and height.

        Optional inputs are:

          perspective_angle - perspective angle of camera, for reporting flow in meters per second
          move_step           - step size in pixels for sampling the flow image
          window_name       - window name for display
          flow_color_rgb    - color for displaying flow
        '''

        self.move_step = move_step
        self.mv_color_bgr = (flow_color_rgb[2], flow_color_rgb[1],
                             flow_color_rgb[0])

        self.perspective_angle = perspective_angle

        self.window_name = window_name

        self.size = (int(frame_width / scaledown),
                     int(frame_height / scaledown))

        self.prev_gray = None
        self.prev_time = None

    def processBytes(self, rgb_bytes, distance=None, timestep=1):
        '''
        Processes one frame of RGB bytes, returning summed X,Y flow.

        Optional inputs are:

          distance - distance in meters to image (focal length) for returning flow in meters per second
          timestep - time step in seconds for returning flow in meters per second
         '''

        frame = np.frombuffer(rgb_bytes, np.uint8)
        frame = np.reshape(frame, (self.size[1], self.size[0], 3))
        return self.processFrame(frame, distance, timestep)

    def processFrame(self, frame, distance=None, timestep=1):
        '''
        Processes one image frame, returning summed X,Y flow and frame.

        Optional inputs are:

          distance - distance in meters to image (focal length) for returning flow in meters per second
          timestep - time step in seconds for returning flow in meters per second
        '''

        frame2 = cv2.resize(frame, self.size)

        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        xsum, ysum = 0, 0

        xvel, yvel = 0, 0

        flow = None

        if not self.prev_gray is None:

            flow = cv2.calcOpticalFlowFarneback(self.prev_gray,
                                                gray,
                                                flow,
                                                pyr_scale=0.5,
                                                levels=5,
                                                winsize=13,
                                                iterations=10,
                                                poly_n=5,
                                                poly_sigma=1.1,
                                                flags=0)
            for y in range(0, flow.shape[0], self.move_step):

                for x in range(0, flow.shape[1], self.move_step):
                    fx, fy = flow[y, x]
                    xsum += fx
                    ysum += fy

            # Default to system time if no timestep
            curr_time = time.time()
            if not timestep:
                timestep = (curr_time -
                            self.prev_time) if self.prev_time else 1
            self.prev_time = curr_time

            xvel = self._get_velocity(flow, xsum, flow.shape[1], distance,
                                      timestep)
            yvel = self._get_velocity(flow, ysum, flow.shape[0], distance,
                                      timestep)

        self.prev_gray = gray

        # Return x,y velocities and new image with flow lines
        return xvel, yvel

    def _get_velocity(self, flow, sum_velocity_pixels, dimsize_pixels,
                      distance_meters, timestep_seconds):

        count = (flow.shape[0] * flow.shape[1]) / self.move_step**2

        average_velocity_pixels_per_second = sum_velocity_pixels / count / timestep_seconds

        return self._velocity_meters_per_second(average_velocity_pixels_per_second, dimsize_pixels, distance_meters) \
            if self.perspective_angle and distance_meters \
            else average_velocity_pixels_per_second

    def _velocity_meters_per_second(self, velocity_pixels_per_second,
                                    dimsize_pixels, distance_meters):

        distance_pixels = (dimsize_pixels / 2) / math.tan(
            self.perspective_angle / 2)

        pixels_per_meter = distance_pixels / distance_meters

        return velocity_pixels_per_second / pixels_per_meter

def takeSecond(elem):
    return elem[1]


import os
if __name__ == "__main__":
    k_num = 10

    src_dir = "/data/shanyx/SLR/Crop"
    des_dir = "/data/shanyx/SLR/Crop_keyframe"

    for ii in range(500):
        print(ii)
        for jj in range(50):
            data_path = os.path.join(src_dir, str(ii), str(jj)+".avi")
            out_path = os.path.join(des_dir, str(ii), str(jj)+".avi")
            videoFile = cv2.VideoCapture(data_path)
            width = int(videoFile.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(videoFile.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps, z = videoFile.get(5), 0
            fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            videoOutputFile = cv2.VideoWriter(out_path, fourcc, fps, (570, 570))
            flow = OpticalFlowCalculator(width,
                                        height,
                                        # window_name='Optical Flow',
                                        scaledown=1,
                                        move_step=16)

            ret, frame = videoFile.read()
            data = []
            plt.cla()
            count = 0
            vals = []
            frames = []
            while ret:
                xvel, yvel = flow.processFrame(frame)
                val = xvel * xvel + yvel * yvel
                if val < 0.002:
                    vals.append((val,count))
                frames.append(frame)
                data.append(math.log(xvel * xvel + yvel * yvel + 1))
                ret, frame = videoFile.read()
                count += 1

            kmeans = KMeans(n_clusters=k_num)
            kmeans.fit(vals)
            kmeans_y_predict = kmeans.predict(vals)
            # print('frame:', len(frames))

            indexs = [[] for k in range(k_num) ]
            final = []
            for k in range(k_num):
                min_vals = []
                min_coun = []
                for index, num in enumerate(kmeans_y_predict.tolist()):
                    if num == k:
                        indexs[k].append(index)
                for index in indexs[k]:
                    va = vals[index][0]
                    cou = vals[index][1]
                    min_vals.append(va)
                    min_coun.append(cou)
                min_index = min_coun[min_vals.index(min(min_vals))]
                final.append((frames[min_index], min_index))

            final.sort(key=takeSecond)
            txt_path = os.path.join(des_dir, str(ii), str(jj) + ".txt")
            with open(txt_path, "w") as f:
                for fin, fi_index in final:
                    videoOutputFile.write(fin)
                    f.write(str(fi_index))
                    f.write("\n")
                    #cv2.imwrite('img' + "/" + str(i) + '/' + str(fi_index) + '.jpg', fin)
            videoOutputFile.release()