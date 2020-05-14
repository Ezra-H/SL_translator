import os
import cv2

src_dir = "H:/Crop"
des_dir = "H:/Merge"

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
fps = 30

for ii in range(500):
    dir1_src = os.path.join(src_dir, str(ii))
    saveVideo_dir = os.path.join(des_dir, str(ii) + ".avi")
    savetxt_dir = os.path.join(des_dir, str(ii) + ".txt")
    out = cv2.VideoWriter(saveVideo_dir, fourcc, fps, (570, 570))
    txtFile = open(savetxt_dir, "w")
    for jj in range(50):
        srcVideo_dir = os.path.join(dir1_src, str(jj)+".avi")
        video = cv2.VideoCapture(srcVideo_dir)
        frameCounts = int(video.get(7))
        print(frameCounts, file=txtFile)
        for kk in range(frameCounts):
            _, frame = video.read()
            out.write(frame)
    txtFile.close()
    out.release()
    print("[%d/500]Running....."%(ii+1))


