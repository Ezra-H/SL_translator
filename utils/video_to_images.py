if __name__ == "__main__":
    import cv2
    vc=cv2.VideoCapture("C:/Users/Administrator/Desktop/2.avi")
    c=1
    if vc.isOpened():
        rval, frame=vc.read()
    else:
        rval=False
    while rval:
        rval, frame=vc.read()
        cv2.imwrite('C:/Users/Administrator/Desktop/2/'+str(c).zfill(2)+'.jpg', frame)
        c=c+1
        cv2.waitKey(1)
    vc.release()