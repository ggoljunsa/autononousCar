import newxhat as hw
import cv2
import newconfig as cfg

img = cv2.imread("images.png")
cv2.imshow("image", img)
start_flag = False

while True:
    k = cv2.waitKey(5)
    if k != -1:
        print(k)

    if k == ord("q"):
        break

    if k == ord("s"):
        if start_flag == False:
            start_flag = True
        else:
            start_flag = False
        print("start_flag: ", start_flag)

    if start_flag == True:
        # Left arrow: 81, Right arrow: 83, Up arrow: 82, Down arrow: 84
        if k == 81:
            hw.motor_one_speed(cfg.maxturn_speed)
            hw.motor_two_speed(cfg.minturn_speed)
            hw.motor_three_speed()
            # print('Straight')
            cfg.wheel = 1
        if k == 83:
            hw.motor_one_speed(cfg.minturn_speed)
            hw.motor_two_speed(cfg.maxturn_speed)
            cfg.wheel = 3
        if k == 82:
            hw.motor_one_speed(cfg.normal_speed_right)
            hw.motor_two_speed(cfg.normal_speed_left)
            cfg.wheel = 2
    else:
        hw.motor_one_speed(0)


hw.motor_clean()
cv2.destroyAllWindows()
