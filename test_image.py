import os
import time
import numpy as np
import torch
import cv2
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

def rotate_and_split_license_plate(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges_image = cv2.Canny(gray_image, 100, 200, apertureSize=3, L2gradient=True)

    # Phát hiện các đoạn thẳng dài và thẳng bằng Hough Line Transform
    lines = cv2.HoughLines(edges_image, 1, np.pi / 180, threshold=100)

    if lines is not None:
        # Lọc và chọn lựa đoạn thẳng phù hợp (đoạn thẳng có độ dài > threshold_length)
        threshold_length = 80
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
        print("filtered_lines:", filtered_lines)
        if len(filtered_lines) > 0:
            # Lựa chọn đoạn thẳng dài nhất
            longest_line = max(filtered_lines, key=lambda x: np.linalg.norm(np.array(x[0]) - np.array(x[1])))
            x1, y1 = longest_line[0]
            x2, y2 = longest_line[1]

            # Tính góc xoay của đường thẳng
            rotation_angle = (np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Xoay lại ảnh để biển số xe nằm ngang
            height, width = image.shape[:2]
            print("height, width:", height, width)
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


if __name__ == "__main__":

    # change to your directory
    image_dir = "./images"
    results_dir = "./results/results_Plate"
    crop_plates_dir = "./results/crop_plates"

    # Load model
    model = torch.load("LP_model_9616images_100e.pth")
    model.eval()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Sử dụng torchvision.transforms để chuẩn bị ảnh
    transform = transforms.Compose([
        transforms.ToTensor(),  # Chuyển đổi ảnh sang tensor
    ])

    detected_plates = 0
    is_Plate = False
    start = time.time()
    for file in os.listdir(image_dir):
        try:
            print("File name:", file)
            image = cv2.imread(os.path.join(image_dir, file))

            # Chuyển đổi ảnh và thêm một chiều batch
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                prediction = model(image_tensor)
                print(prediction[0])

            # Chuyển tensor sang CPU và numpy array
            boxes = prediction[0]['boxes'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()

            # Lấy bounding box
            for box, score in zip(boxes, scores):
                if score > 0.8:
                    is_Plate = True
                    print("detected_plates:", detected_plates)
                    x_min, y_min, x_max, y_max = map(int, box)
                    width = x_max-x_min
                    height = y_max-y_min

                    if (width)/(height) <= 1.75:  # width / height, nếu biển số hơi vuông
                        print("Short Plate!")
                        img1 = np.copy(image[int(y_min):int(y_max), int(x_min):int(x_max)])
                        image_upper, image_lower = rotate_and_split_license_plate(img1)
                        print("Hey")

                        if image_upper is None and image_lower is None:
                            split_point = y_min + (y_max - y_min) // 2

                            # Tạo hai phần ảnh con từ ảnh cr_img
                            upper_part = image[int(y_min):int(split_point), int(x_min):int(x_max)]
                            lower_part = image[int(split_point):int(y_max), int(x_min):int(x_max)]

                            # Điều chỉnh chiều cao của cả hai phần ảnh để chúng có cùng chiều cao
                            image_upper = cv2.resize(upper_part, (int(width), int(height / 2)))
                            image_lower = cv2.resize(lower_part, (int(width), int(height / 2)))

                        image_collage_horizontal = np.hstack([image_upper, image_lower])
                        # Lưu plate đã cắt đôi vào thư mục crop_plates
                        plate_path = os.path.join(crop_plates_dir, file)
                        cv2.imwrite(plate_path, image_collage_horizontal)
                    else:  # nếu là biển số dài
                        print("Long Plate!")
                        # Lưu plate đã detect vào thư mục crop_plates
                        plate = image[y_min:y_max, x_min:x_max]  # Cắt và lưu phần plate đã detect
                        plate_path = os.path.join(crop_plates_dir, file)
                        cv2.imwrite(plate_path, plate)

                    # Vẽ bounding box lên ảnh
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(image, f'{score:.4f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)

                    # break  # Kết thúc vòng lặp để chỉ lấy 1 biển số duy nhất
            if is_Plate:
                detected_plates += 1
                # Lưu ảnh vào thư mục results
                output_path = os.path.join(results_dir, file)
                cv2.imwrite(output_path, image)
                print(f"Saved output image with bounding box to {output_path}")
            is_Plate = False

        except Exception as e:
            print("error:", e)
            continue

    end = time.time()
    print("Average time per frame:", (end-start)/len(os.listdir(image_dir)))
    print("Detected Plates:", detected_plates)
