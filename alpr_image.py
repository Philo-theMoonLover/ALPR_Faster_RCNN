import os
import time
import numpy as np
import torch
import cv2
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt



def model_predict(model, image):
    with torch.no_grad():
        prediction = model(image)
        # print(prediction[0])
    # Chuyển tensor sang CPU và numpy array
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()

    return boxes, scores, labels

def rotate_and_split_license_plate(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges_image = cv2.Canny(gray_image, 100, 200, apertureSize=3, L2gradient=True)

    # Phát hiện các đoạn thẳng dài và thẳng bằng Hough Line Transform
    lines = cv2.HoughLines(edges_image, 1, np.pi / 180, threshold=100)

    if lines is not None:
        # Lọc và chọn lựa đoạn thẳng phù hợp (đoạn thẳng có độ dài > threshold_length)
        threshold_length = 100
        filtered_lines = []
        for line in lines:
            rho, theta = line[0]
            if np.pi / 4 < theta < 3 * np.pi / 4:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * a)
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if line_length > threshold_length:
                    filtered_lines.append(((x1, y1), (x2, y2)))

                # Vẽ các đường thẳng
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if filtered_lines:
            # Lựa chọn đoạn thẳng dài nhất
            longest_line = max(filtered_lines, key=lambda x: np.linalg.norm(np.array(x[0]) - np.array(x[1])))
            x1, y1 = longest_line[0]
            x2, y2 = longest_line[1]

            # Tính góc xoay của đường thẳng
            rotation_angle = (np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Xoay lại ảnh để biển số xe nằm ngang
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

            print("FOUND LINES.")
            print("rotate angle:", rotation_angle)
        else:
            # plt.figure(figsize=(12, 8))
            # plt.subplot(1, 2, 1), plt.imshow(edges_image, cmap='gray')
            # plt.title('Canny Edges'), plt.xticks([]), plt.yticks([])
            # plt.subplot(1, 2, 2), plt.imshow(image, cmap='gray')
            # plt.title('HoughLines Transform'), plt.xticks([]), plt.yticks([])
            # plt.show()
            return None, None

        # Tính toán điểm chia ảnh thành hai phần trên và dưới
        split_point = height // 2

        # Tạo hai phần ảnh con từ ảnh rotated_image
        upper_part = rotated_image[:split_point, :]
        lower_part = rotated_image[split_point:, :]
        # print("upper_part:", upper_part.shape)
        # print("lower_part", lower_part.shape)

        # Điều chỉnh chiều cao của cả hai phần ảnh để chúng có cùng chiều cao
        upper_part = cv2.resize(upper_part, (int(width), int(height / 2)))
        lower_part = cv2.resize(lower_part, (int(width), int(height / 2)))

        return upper_part, lower_part
    else:
        print("NO LINES!")
        return None, None

# Tạo hàm ánh xạ từ ID sang nhãn
def id_to_label(id):
    if 1 <= id <= 10:
        return str(id - 1)  # Từ 1-10 ánh xạ đến '0'-'9'
    elif 11 <= id <= 36:
        return chr(id - 11 + ord('A'))  # Từ 11-36 ánh xạ đến 'A'-'Z'
    else:
        return 'unknown'

if __name__ == "__main__":

    # change to your directory
    image_dir = "./images"
    results_dir = "./results/results_ALPR"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    threshold_plate = 0.8
    threshold_ocr = 0.75

    model_LP_detect = torch.load("LP_model_9616images_100e.pth")
    model_OCR = torch.load("LP_OCR_model_60e.pth")
    model_LP_detect.eval()
    model_OCR.eval()

    model_LP_detect.to(device)
    model_OCR.to(device)

    # Sử dụng torchvision.transforms để chuẩn bị ảnh
    transform = transforms.Compose([
        transforms.ToTensor(),  # Chuyển đổi ảnh sang tensor
    ])

    start = time.time()
    for file in os.listdir(image_dir):
        try:
            print("File name:", file)
            image = cv2.imread(os.path.join(image_dir, file))
            # Chuyển đổi ảnh và thêm một chiều batch
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Detect_Plate
            boxes_plate, scores_plate, _ = model_predict(model_LP_detect, image_tensor)

            # Chỉ lấy bounding box đầu tiên
            for box, score in zip(boxes_plate, scores_plate):
                if score > threshold_plate:
                    x_min, y_min, x_max, y_max = map(int, box)
                    width = x_max - x_min
                    height = y_max - y_min

                    if width / height <= 1.75:  # width / height, nếu ko phải biển số dài
                        img1 = np.copy(image[int(y_min):int(y_max), int(x_min):int(x_max)])
                        image_upper, image_lower = rotate_and_split_license_plate(img1)

                        if image_upper is None and image_lower is None:
                            split_point = y_min + (y_max - y_min) // 2

                            # Tạo hai phần ảnh con từ ảnh cr_img
                            upper_part = image[int(y_min):int(split_point), int(x_min):int(x_max)]
                            lower_part = image[int(split_point):int(y_max), int(x_min):int(x_max)]

                            # Điều chỉnh chiều cao của cả hai phần ảnh để chúng có cùng chiều cao
                            image_upper = cv2.resize(upper_part, (int(width), int(height / 2)))
                            image_lower = cv2.resize(lower_part, (int(width), int(height / 2)))

                        crop_plate = np.hstack([image_upper, image_lower])
                    else:
                        crop_plate = image[int(y_min):int(y_max), int(x_min):int(x_max)]
                    # Vẽ bounding box lên ảnh
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(image, f'Score: {score:.4f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)
                    # break  # Kết thúc vòng lặp để chỉ lấy 1 biển số duy nhất


            # Predict Characters
            ocr_tensor = transform(crop_plate).unsqueeze(0).to(device)
            boxes_ocr, scores_ocr, labels_ocr = model_predict(model_OCR, ocr_tensor)

            boxes_info = []
            # Lặp qua tất cả các bounding box và scores
            for box_ocr, score_ocr, label_ocr in zip(boxes_ocr, scores_ocr, labels_ocr):
                if score_ocr > threshold_ocr:
                    boxes_info.append((box_ocr, score_ocr, label_ocr))

            boxes_info = sorted(boxes_info, key=lambda x: x[0][0])  # Sắp xếp lại ký tự theo trục Ox
            print("boxes_info:", boxes_info)
            labels_list = [id_to_label(i[2]) for i in boxes_info]
            if not labels_list:
                continue

            labels_text = ''.join(labels_list)
            # Ghi ký tự biển số lên ảnh
            cv2.putText(image, labels_text, (x_min, y_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # # Với bộ ảnh có nhiều plate trong ảnh
            # plate_count = 0
            # # Lặp qua tất cả các bounding box và scores
            # for box, score in zip(boxes, scores):
            #     if score > 0.8:
            #         x_min, y_min, x_max, y_max = map(int, box)
            #
            #         # Lưu plate đã detect vào thư mục crop_plates
            #         plate = image[y_min:y_max, x_min:x_max]  # Cắt và lưu phần plate đã detect
            #         plate_count += 1
            #         if plate_count == 1:
            #             plate_file_name = f"{os.path.splitext(file)[0]}.jpg"
            #         else:
            #             plate_file_name = f"{os.path.splitext(file)[0]}_{plate_count}.jpg"
            #         plate_path = os.path.join(crop_plates_dir, plate_file_name)
            #         cv2.imwrite(plate_path, plate)
            #
            #         # Vẽ bounding box lên ảnh
            #         cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            #         cv2.putText(image, f'Score: {score:.4f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            #                     (0, 255, 0), 2)

            # Lưu ảnh vào thư mục results_ALPR
            output_path = os.path.join(results_dir, file)
            cv2.imwrite(output_path, image)
            print(f"Saved output image with bounding box to {output_path}")
        except Exception as e:
            continue

    print("Average time per image:", (time.time() - start) / len(os.listdir(image_dir)))


