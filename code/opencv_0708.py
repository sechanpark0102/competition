import numpy as np
import cv2
import math
import Function_Library as fl
import serial
import time

EPOCH = 500000

arduino = serial.Serial('COM3', baudrate= 115200, timeout=5)
time.sleep(2)

def angle_transform(x):
    if x > 0:
        if x < 70:
            return -20
        else:
            return (x - 90)
    else:
        if x > -70:
            return 20
        else:
            return (x + 90)

def pt(x):
    a = (-x / 40) * 140 + 500
    return a


# 파이 시리얼 통신 코드
def py_serial(line_angle_input):
    # 음수 값이면 '-'를 붙여서 전송
    # if float(line_angle_input) < 0:
    #     line_angle_input = "-" + line_angle_input

    # 숫자 두개를 하나의 문자열로 변환하여 통신(오버헤드 줄이려면 한 문자열로 하는게 좋음)
    message = f"{line_angle_input}\n"
    arduino.write(message.encode())
    # 아두이노로부터 입력 받은 응답을 버퍼를 비우기 위해 읽어옴
    # while arduino.in_waiting > 0:
    #     response = arduino.readline().decode().strip()
    #     print(response)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255  # <-- This line altered for grayscale.

    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
def bird2 (image):
    # targeted rectangle on original image which needs to be transformed
    tl = [100, 240]  # 좌상
    tr = [540, 240]  # 좌하
    br = [940, 480]  # 우상
    bl = [-300, 480]  # 우하
    corner_points_array = np.float32([tl,tr,br,bl])

    # original image dimensions
    width = 640
    height = 480

    # Create an array with the parameters (the dimensions) required to build the matrix
    imgTl = [0,0]
    imgTr = [width,0]
    imgBr = [width,height]
    imgBl = [0,height]
    img_params = np.float32([imgTl,imgTr,imgBr,imgBl])

    # Compute and return the transformation matrix
    matrix = cv2.getPerspectiveTransform(corner_points_array,img_params)
    img_transformed = cv2.warpPerspective(image,matrix,(width,height))
    return img_transformed

def bird_eye(image):
    p1 = [210, 240]  # 좌상
    p2 = [0, 480]  # 좌하
    p3 = [400, 240]  # 우상
    p4 = [640, 480]  # 우하

    # corners_point_arr는 변환 이전 이미지 좌표 4개
    corner_points_arr = np.float32([p1, p2, p3, p4])
    height, width = image.shape[:2]

    image_p1 = [0, 0]
    image_p2 = [width, 0]
    image_p3 = [width, height]
    image_p4 = [0, height]

    image_params = np.float32([image_p1, image_p2, image_p3, image_p4])

    mat = cv2.getPerspectiveTransform(corner_points_arr, image_params)
    # mat = 변환행렬(3*3 행렬) 반
    image_transformed = cv2.warpPerspective(image, mat, (width, height))
    return image_transformed

def draw_lines(img, lines, color=[0, 0, 255], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img


if __name__ == "__main__":
    # Exercise Environment Setting
    env = fl.libCAMERA()
    # Camera Initial Setting
    ch1, ch2 = env.initial_setting(capnum=2)
    # Camera Reading..
    for i in range(EPOCH):
        _, frame0, _, frame1 = env.camera_read(ch1, ch2)
        # 배태현 작성
        height = 480
        width = 640

        region_of_interest_vertices = [
            (0, height),
            (width / 2 - 250, height / 2 - 80),
            (width / 2 + 250, height / 2 - 80),
            (width, height),
        ]   #자르는 모양 조절

        # print(frame1.shape)

        bird345=bird2(frame1)
        gray_image = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        cannyed_image = cv2.Canny(gray_image, 150, 300)
        # Moved the cropping operation to the end of the pipeline.
        # cropped_image = region_of_interest(
        #     cannyed_image,
        #     np.array([region_of_interest_vertices], np.int32)
        # )
        bird=bird2(cannyed_image)

        # 변수 조절해야함
        lines = cv2.HoughLinesP(
            bird,
            rho=6,
            theta=np.pi / 60,
            threshold=80,  # 낮을수록 선 더 잘 검출함
            lines=np.array([]),
            minLineLength=55,  # 검출할 선의 최소길이
            maxLineGap=25
        )
        # print(lines)


        line_image = draw_lines(bird345, lines)


        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []

        min_y = frame1.shape[0] * (1 / 5)  # <-- Just below the horizon 원하는 선 길이로 조절
        min_y = round(min_y)
        max_y = frame1.shape[0]  # <-- The bottom of the image

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x1 == x2:
                        continue
                    slope = (y2 - y1) / (x2 - x1)  # <-- Calculating the slope.
                    if math.fabs(slope) < 0.5:  # <-- Only consider extreme slope
                        continue
                    if x2<320 and x1<320 :  # <-- If the slope is negative, left group.
                        left_line_x.extend([x1, x2])
                        left_line_y.extend([y1, y2])
                    elif x1>=320 and x2>=320:  # <-- Otherwise, right group.
                        right_line_x.extend([x1, x2])
                        right_line_y.extend([y1, y2])
                    else : pass
            if left_line_x == []:
                left_line_x.extend([100, 200])
            if left_line_y == []:
                left_line_y.extend([480, 240])
            if right_line_x == []:
                right_line_x.extend([540, 440])
            if right_line_y == []:
                right_line_y.extend([480, 240])
        else:
            left_line_x.extend([100, 200])
            left_line_y.extend([480, 240])
            right_line_x.extend([540, 440])
            right_line_y.extend([480, 240]) # 선 검출안되면 임의로 넣어주는값

        # print(left_line_x)
        # print(left_line_y)
        # print(right_line_x)
        # print(right_line_y)

        poly_left = np.poly1d(np.polyfit(
            left_line_y,
            left_line_x,
            deg=1
        ))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))

        poly_right = np.poly1d(np.polyfit(
            right_line_y,
            right_line_x,
            deg=1
        ))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))
        center_x_up=(left_x_end+right_x_end)/2
        center_x_down=(left_x_start+right_x_start)/2
        center_x_down=round(center_x_down)
        center_x_up=round(center_x_up)

        center_slope = (max_y - min_y) / (center_x_down - center_x_up) if (center_x_down - center_x_up) != 0 else 0
        center_angle = np.degrees(np.arctan(center_slope))

        #조향각 가변저항에 할당 전송
        abc = angle_transform(center_angle)
        aabbcc = pt(abc)

        new_line = [
            [[left_x_start, max_y, left_x_end, min_y]],
            [[right_x_start, max_y, right_x_end, min_y]],
            [[center_x_down, max_y, center_x_up, min_y]]
        ]

        line_image_final = draw_lines(
            bird345,
            new_line,
            thickness=5,
        )

        cv2.putText(line_image_final, f"Center Angle: {aabbcc:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # env.image_show(bird) #cannyed 확인

        # env.image_show(cropped_image) #crop 확인
        # env.image_show(bird) #crop 확인

        # if lines is not None:             #line들 확인
        #     env.image_show(line_image)    #line들 확인
        # else:                             #line들 확인
        #     env.image_show(frame1)        #line들 확인

        env.image_show(line_image_final,line_image)  #최종 line 확인
        # 파이시리얼 통신 (현장에서 angle, distance_to_right_line 변수 확인 및 튜닝 필요)
        py_serial(aabbcc)



        # cropped_image_test = region_of_interest(
        #     gray_image,
        #     np.array([region_of_interest_vertices], np.int32)
        # )
        # env.image_show(cropped_image_test) # 화면 자르는거 테스트용


        if env.loop_break():
            break
