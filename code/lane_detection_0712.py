import numpy as np
import cv2
import math
import Function_Library as fl
import serial
import time
import function_by_bt as bt
import matplotlib.pyplot as plt
EPOCH = 500000
arduino = serial.Serial('COM3', baudrate=115200, timeout=5)
time.sleep(2)

def py_serial(line_angle_input,dist_left,dist_right):
    # 음수 값이면 '-'를 붙여서 전송
    # if float(line_angle_input) < 0:
    #     line_angle_input = "-" + line_angle_input

    # 숫자 두개를 하나의 문자열로 변환하여 통신(오버헤드 줄이려면 한 문자열로 하는게 좋음)
    message = f"{line_angle_input},{dist_left},{dist_right}\n"
    arduino.write(message.encode())
    # 아두이노로부터 입력 받은 응답을 버퍼를 비우기 위해 읽어옴
    # while arduino.in_waiting > 0:
    #     response = arduino.readline().decode().strip()
    #     print(response)

def sliding_window_search(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = np.int32(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 30
    window_height = np.int32(binary_warped.shape[0] // nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 50
    minpix = 70
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


if __name__ == "__main__":
    # Exercise Environment Setting
    env = fl.libCAMERA()
    # Camera Initial Setting
    ch1, ch2 = env.initial_setting(capnum=2)
    # Camera Reading..
    for i in range(EPOCH):
        _, frame1, _, frame0 = env.camera_read(ch1, ch2)
        height = 448
        width = 800
        region_of_interest_vertices = [
            (210, 300),
            (590, 300),
            (680, 448),
            (120, 448)  ]  # 자르는 모양 조절


        # 카메라 calibration
        bird234 = bt.bird2(frame1)
        clhae=bt.clahe_equalization(frame1)
        cropped_image = bt.region_of_interest(
            frame1,
            np.array([region_of_interest_vertices], np.int32)
        )
        rgb = clhae
        hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        # Create a mask for the condition: V >= 200 and R, G, B >= 200
        v_mask = hsv[:, :, 2] >= 200
        r_mask = rgb[:, :, 2] >= 200
        g_mask = rgb[:, :, 1] >= 200
        b_mask = rgb[:, :, 0] >= 200
        # Combine masks
        final_mask = v_mask & r_mask & g_mask & b_mask
        # Create the output image
        output_img = np.zeros_like(rgb)
        output_img[final_mask] = [255, 255, 255]  # Set to white where the mask is True
        # cropped_image = bt.region_of_interest(
        #     bird,
        #     np.array([region_of_interest_vertices], np.int32)
        # )

        # filtered = bt.color_filter(frame1)
        bird = bt.bird2(output_img)
        gray = cv2.cvtColor(bird, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        # # adaptive_thresh = cv2.adaptiveThreshold(cropped_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # # canny=cv2.Canny(thresh,245,400)
        # # filtered = bt.color_filter(frame1)
        # # filtered_BRG=cv2.cvtColor(filtered, cv2.COLOR_HSV2BGR)
        # # gauss = cv2.GaussianBlur(filtered, kernel_size, 0)
        # # gray_image = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        # # morp_image = env.morphology(gray_image, (3, 3), mode='closing')
        # # cannyed_image = cv2.Canny(gray_image, 245, 400)
        # # bird = bt.bird2(filtered)
        #
        leftx, lefty, rightx, righty, out_img = sliding_window_search(thresh)

        if len(leftx) > 0 and len(lefty) > 0 and len(rightx) > 0 and len(righty) > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            ploty = np.linspace(0, thresh.shape[0] - 1, thresh.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]

            for i in range(len(ploty)):
                cv2.circle(out_img, (int(left_fitx[i]), int(ploty[i])), 2, (0, 255, 255), -1)
                cv2.circle(out_img, (int(right_fitx[i]), int(ploty[i])), 2, (0, 255, 255), -1)

            center_fitx = (left_fitx + right_fitx) / 2
            center_y = ploty
            center_points = np.array([np.transpose(np.vstack([center_fitx, center_y]))])
            cv2.polylines(out_img, np.int32([center_points]), isClosed=False, color=(255, 255, 0), thickness=2)

            # 차선의 중심 위치 계산 및 각도 계산
            center_x_down = int(center_fitx[-1])
            center_x_up = int(center_fitx[0])
            center_slope = (center_y[-1] - center_y[0]) / (center_x_down - center_x_up) if (center_x_down - center_x_up) != 0 else 0
            center_angle = np.degrees(np.arctan(center_slope))
            # 조향각 가변저항에 할당 전송
            abc = bt.angle_transform(center_angle)
            aabbcc = bt.pt(abc)
            cv2.putText(out_img, f"Center Angle: {aabbcc:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            left_center = [300, 410]
            right_center = [500, 410]
            cv2.circle(out_img, left_center, 10, (0, 255, 0), 4)
            cv2.circle(out_img, right_center, 10, (0, 255, 0), 4)
            left_circle = [int(left_fitx[410]),420]
            right_circle = [int(right_fitx[410]),420]
            cv2.circle(out_img, left_circle, 10, (0, 255, 255), 4)
            cv2.circle(out_img, right_circle, 10, (0, 255, 255), 4)
            center_circle = [center_x_down, 460]
            cv2.circle(out_img, center_circle, 10, (255, 0, 0), 4)
            dist = left_center[0] - center_circle[0]
            # cv2.putText(out_img, f"Dist: {dist:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            dist_left=left_center[0] - left_circle[0]
            dist_right=right_circle[0] - right_center[0]
            aabbcc=bt.dist(dist_left,dist_right,aabbcc)
            cv2.putText(out_img, f"Dist-L: {dist_left:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(out_img, f"Dist-R: {dist_right:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else :
            center_angle = 0
            dist = 0
            abc = 0

        env.image_show(out_img,thresh)  # 최종 line 확인
        py_serial(aabbcc,dist_left,dist_right)

        response = arduino.readline()
        response=response[:len(response) - 1].decode()
        print(f"Sent: {aabbcc}, Received: {response}")

        if env.loop_break():
            break
