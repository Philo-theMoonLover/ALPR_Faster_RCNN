import os
import time

import torch
import cv2
import torchvision
from torchvision import transforms



# Tạo hàm ánh xạ từ ID sang nhãn
def id_to_label(id):
    if 1 <= id <= 10:
        return str(id - 1)  # Từ 1-10 ánh xạ đến '0'-'9'
    elif 11 <= id <= 36:
        return chr(id - 11 + ord('A'))  # Từ 11-36 ánh xạ đến 'A'-'Z'
    else:
        return 'unknown'


if __name__ == "__main__":

    # Images directory
    image_dir = "D:/ALPR_Collections/ALPR_Faster_RCNN/crop_plates"
    results_dir = "D:/ALPR_Collections/ALPR_Faster_RCNN/results_OCR"
    model = torch.load("LP_OCR_model_60e.pth")
    model.eval()

    # print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU to train
    model.to(device)

    # Sử dụng torchvision.transforms để chuẩn bị ảnh
    transform = transforms.Compose([
        transforms.ToTensor(),  # Chuyển đổi ảnh sang tensor
    ])

    start = time.time()
    for file in os.listdir(image_dir):
        try:
            image = cv2.imread(os.path.join(image_dir, file))

            # Chuyển đổi ảnh và thêm một chiều batch
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                prediction = model(image_tensor)
                # print(prediction[0])

            # Chuyển tensor sang CPU và numpy array
            boxes = prediction[0]['boxes'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()
            labels = prediction[0]['labels'].cpu().numpy()

            box_info = []
            # Lặp qua tất cả các bounding box và scores
            for box, score, label in zip(boxes, scores, labels):
                if score > 0.6:
                    box_info.append((box, score, label))
                    # Vẽ bounding box lên ảnh
                    x_min, y_min, x_max, y_max = map(int, box)
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    # cv2.putText(image, f'Score: {score:.4f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    #             (0, 255, 0), 2)
                    # label_text = f'{id_to_label(label)}'
                    # cv2.putText(image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)

            box_info_sorted = sorted(box_info, key=lambda x: x[0][0])  # Sắp xếp ký tự theo trục Ox
            print(box_info_sorted)
            label_list = [id_to_label(i[2]) for i in box_info_sorted]
            if not label_list:
                continue

            # Tạo tên file mới với nhãn
            labels_text = ''.join(label_list)
            base_name, ext = os.path.splitext(file)
            new_file_name = f"{base_name}_{labels_text}{ext}"
            new_file_path = os.path.join(results_dir, new_file_name)
            # Đảm bảo tên file không bị trùng
            count = 1
            while os.path.exists(new_file_path):
                new_file_name = f"{base_name}_{labels_text}_({count}){ext}"
                new_file_path = os.path.join(results_dir, new_file_name)
                count += 1

            # Lưu ảnh vào thư mục results
            cv2.imwrite(new_file_path, image)
            print(f"Saved output image with bounding box to {new_file_path}")
        except Exception as e:
            continue

    end = time.time()
    print("Average time per frame:", (end-start)/len(os.listdir(image_dir)))

