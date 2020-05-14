import os
import cv2
if __name__ == "__main__":
    src_dir = "C:/Users/Administrator/Desktop/31May_2011_Tuesday_tagesschau-4302"
    saveVideo_dir = "C:/Users/Administrator/Desktop/31May_2011_Tuesday_tagesschau-4302/final.avi"

    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    fps = 30
    out = cv2.VideoWriter(saveVideo_dir, fourcc, fps, (210, 260))
    for jj in range(110):
        image_dir = os.path.join(src_dir, "images"+str(jj).zfill(4)+".png")
        image = cv2.imread(image_dir)
        out.write(image)
    out.release()


