import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import math


class OpticalFlowCalculator:
    '''
    A class for optical flow calculations using OpenCV
    '''

    def __init__(self, frame_width, frame_height, scaledown=1,
                 perspective_angle=0, move_step=16, window_name=None, flow_color_rgb=(0, 255, 0)):
        '''
        Creates an OpticalFlow object for images with specified width and height.

        Optional inputs are:

          perspective_angle - perspective angle of camera, for reporting flow in meters per second
          move_step           - step size in pixels for sampling the flow image
          window_name       - window name for display
          flow_color_rgb    - color for displaying flow
        '''

        self.move_step = move_step
        self.mv_color_bgr = (flow_color_rgb[2], flow_color_rgb[1], flow_color_rgb[0])

        self.perspective_angle = perspective_angle

        self.window_name = window_name

        self.size = (int(frame_width / scaledown), int(frame_height / scaledown))

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

            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, flow, pyr_scale=0.5, levels=5, winsize=13,
                                                iterations=10, poly_n=5, poly_sigma=1.1, flags=0)
            for y in range(0, flow.shape[0], self.move_step):

                for x in range(0, flow.shape[1], self.move_step):
                    fx, fy = flow[y, x]
                    xsum += fx
                    ysum += fy

                    cv2.line(frame2, (x, y), (int(x + fx), int(y + fy)), self.mv_color_bgr)
                    cv2.circle(frame2, (x, y), 1, self.mv_color_bgr, -1)

            # Default to system time if no timestep
            curr_time = time.time()
            if not timestep:
                timestep = (curr_time - self.prev_time) if self.prev_time else 1
            self.prev_time = curr_time

            xvel = self._get_velocity(flow, xsum, flow.shape[1], distance, timestep)
            yvel = self._get_velocity(flow, ysum, flow.shape[0], distance, timestep)

        self.prev_gray = gray

        if self.window_name:
            cv2.imshow(self.window_name, frame2)
            if cv2.waitKey(1) & 0x000000FF == 27:  # ESC
                return None

        # Return x,y velocities and new image with flow lines
        return xvel, yvel, frame2

    def _get_velocity(self, flow, sum_velocity_pixels, dimsize_pixels, distance_meters, timestep_seconds):

        count = (flow.shape[0] * flow.shape[1]) / self.move_step ** 2

        average_velocity_pixels_per_second = sum_velocity_pixels / count / timestep_seconds

        return self._velocity_meters_per_second(average_velocity_pixels_per_second, dimsize_pixels, distance_meters) \
            if self.perspective_angle and distance_meters \
            else average_velocity_pixels_per_second

    def _velocity_meters_per_second(self, velocity_pixels_per_second, dimsize_pixels, distance_meters):

        distance_pixels = (dimsize_pixels / 2) / math.tan(self.perspective_angle / 2)

        pixels_per_meter = distance_pixels / distance_meters

        return velocity_pixels_per_second / pixels_per_meter


if __name__ == "__main__":
    #for ii in range(50):
    #data_path = "C:/Users/Administrator/Desktop/134/"+str(i)+".avi"
    data_path = "C:/Users/Administrator/Desktop/1.avi"
    videoFile = cv2.VideoCapture(data_path)

    width = int(videoFile.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoFile.get(cv2.CAP_PROP_FRAME_HEIGHT))
    flow = OpticalFlowCalculator(width, height, window_name='Optical Flow', scaledown=1, move_step=16)

    ret, frame = videoFile.read()
    data = []
    while ret:
        xvel, yvel, _ = flow.processFrame(frame)
        #img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        print(xvel*xvel+yvel*yvel)
        data.append(math.log(xvel*xvel+yvel*yvel+1))
        ret, frame = videoFile.read()
    print(data)

    x1 = range(len(data))
    plt.plot(x1, data, label='loss_history',
            linewidth=3, color='pink',marker='o',
            markerfacecolor='blue',markersize=5)
    plt.savefig("C:/Users/Administrator/Desktop/1.jpg")
    plt.show()