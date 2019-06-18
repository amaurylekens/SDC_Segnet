import cv2

while True:
    img=cv2.imread("frame.png")
    cv2.imshow("stream",img)
    k=cv2.waitKey(10) & 0XFF