import cv2

while True:
    k = cv2.waitKey(5)
    print(k)
    if k == ord("s"):
        print("finish")
        break

