import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import Function_Library as fl

EPOCH = 500000


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255  # <-- This line altered for grayscale.

    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

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
            (width / 2 - 140, height / 2),
            (width / 2 + 140, height / 2),
            (width, height),
        ]   #자르는 모양 조절

        # print(frame1.shape)


        gray_image = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        cannyed_image = cv2.Canny(gray_image, 100, 200)
        # Moved the cropping operation to the end of the pipeline.
        cropped_image = region_of_interest(
            cannyed_image,
            np.array([region_of_interest_vertices], np.int32)
        )


        # 변수 조절해야함
        lines = cv2.HoughLinesP(
            cropped_image,
            rho=6,
            theta=np.pi / 60,
            threshold=80,  # 낮을수록 선 더 잘 검출함
            lines=np.array([]),
            minLineLength=100,  # 검출할 선의 최소길이
            maxLineGap=25
        )
        # print(lines)


        line_image = draw_lines(frame1, lines)


        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []

        min_y = frame1.shape[0] * (3 / 5)  # <-- Just below the horizon 원하는 선 길이로 조절
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
                    if slope <= 0:  # <-- If the slope is negative, left group.
                        left_line_x.extend([x1, x2])
                        left_line_y.extend([y1, y2])
                    else:  # <-- Otherwise, right group.
                        right_line_x.extend([x1, x2])
                        right_line_y.extend([y1, y2])
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

        new_line = [
            [[left_x_start, max_y, left_x_end, min_y]],
            [[right_x_start, max_y, right_x_end, min_y]],
        ]

        line_image_final = draw_lines(
            frame1,
            new_line,
            thickness=5,
        )

        # env.image_show(cannyed_image) #cannyed 확인

        # env.image_show(cropped_image) #crop 확인

        # if lines is not None:             #line들 확인
        #     env.image_show(line_image)    #line들 확인
        # else:                             #line들 확인
        #     env.image_show(frame1)        #line들 확인

        env.image_show(line_image_final)  #최종 line 확인


        # cropped_image_test = region_of_interest(
        #     gray_image,
        #     np.array([region_of_interest_vertices], np.int32)
        # )
        # env.image_show(cropped_image_test) # 화면 자르는거 테스트용


        if env.loop_break():
            break
