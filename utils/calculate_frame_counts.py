import cv2
import os
import numpy
if __name__ == "__main__":

    root_dir = "I:/Crop"
    save_txt = "C:/Users/Administrator/Desktop/xf500_color_video/Frame_Counts.npy"
    frame_counts = numpy.empty(shape=(500, 50), dtype=numpy.int)
    for ii in range(500):
        root_in_dir = os.path.join(root_dir, str(ii))
        for jj in range(50):
            video_dir = os.path.join(root_in_dir, str(jj)+".avi")
            cap = cv2.VideoCapture(video_dir)
            frame_counts[ii, jj] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    numpy.save(save_txt, frame_counts)
