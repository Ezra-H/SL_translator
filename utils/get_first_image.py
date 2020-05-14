import os
import cv2
if __name__ == "__main__":
    src_dir = "E:/SLRT/SLR_dataset/xf500_color_video"
    des_dir = "E:/Crop_truth"

    for ii in range(500):
        temp1 = os.path.join(src_dir, str(ii))
        temp1_ = os.path.join(des_dir, str(ii))
        for jj in range(50):
            temp2 = os.path.join(temp1, str(jj)+".avi")
            temp2_ = os.path.join(temp1_, str(jj)+".jpg")
            video = cv2.VideoCapture(temp2)
            video.read()
            _, frame = video.read()
            cv2.imwrite(temp2_, frame)