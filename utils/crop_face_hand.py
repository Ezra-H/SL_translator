import cv2
import os
import json
import numpy

src1_dir = "C:/Users/Administrator/Desktop/srcsrc"
src2_dir = "C:/Users/Administrator/Desktop/openposed"
des_dir = "C:/Users/Administrator/Desktop/SLR_Data_New"
frame_counts_dir = "C:/Users/Administrator/Desktop/xf500_color_video/Frame_Counts.npy"

frame_counts_data = numpy.load(frame_counts_dir)
for ii in range(500):
    for jj in range(1, 50):
        frame_counts_data[ii, jj] = frame_counts_data[ii, jj-1] + frame_counts_data[ii, jj]

for ii in range(3):
    src2_dir_in = os.path.join(src2_dir, str(ii))
    des_dir_in = os.path.join(des_dir, str(ii))

    cap = cv2.VideoCapture(os.path.join(src1_dir, str(ii)+".avi"))
    frame_counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps, z = cap.get(5), 0
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')

    video_dir = os.path.join(des_dir_in, "0")
    cap_face = cv2.VideoWriter(os.path.join(video_dir, "face.avi"), fourcc, fps, (150, 150))
    cap_hand_left = cv2.VideoWriter(os.path.join(video_dir, "hand_left.avi"), fourcc, fps, (150, 150))
    cap_hand_right = cv2.VideoWriter(os.path.join(video_dir, "hand_right.avi"), fourcc, fps, (150, 150))
    for jj in range(frame_counts):

        if (jj > frame_counts_data[ii, z]):
            cap_face.release()
            cap_hand_left.release()
            cap_hand_right.release()
            z += 1
            video_dir = os.path.join(des_dir_in, str(z))
            cap_face = cv2.VideoWriter(os.path.join(video_dir, "face.avi"), fourcc, fps, (150, 150))
            cap_hand_left = cv2.VideoWriter(os.path.join(video_dir, "hand_left.avi"), fourcc, fps, (150, 150))
            cap_hand_right = cv2.VideoWriter(os.path.join(video_dir, "hand_right.avi"), fourcc, fps, (150, 150))

        file_path = os.path.join(src2_dir_in, str(ii)+"_"+str(jj).zfill(12)+"_keypoints.json")
        file = open(file_path)
        data = json.load(file)
        data = data['people'][0]['pose_keypoints_2d']
        data = [int(item) for item in data]
        _, frame = cap.read()
        face_img = frame[data[1] - 75:data[1] + 75, data[0] - 75:data[0] + 75]
        hand_left_img = frame[data[13] - 75:data[13] + 75, data[12] - 75:data[12] + 75]
        hand_right_img = frame[data[22] - 75:data[22] + 75, data[21] - 75:data[21] + 75]

        #cv2.imshow("face", face_img)
        #cv2.imshow("hand_left", hand_left_img)
        #cv2.imshow("hand_right", hand_right_img)
        #cv2.waitKey(0)

        save_dir = os.path.join(des_dir_in, str(z), )
        cap_face.write(face_img)
        cap_hand_left.write(hand_left_img)
        cap_hand_right.write(hand_right_img)