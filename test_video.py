import os
import time

import torch
import cv2
import torchvision
from torchvision import transforms


# change to your directory
vid_dir = "path/to/your/test_video"
video_name = "/test_1.mp4"

results_dir = "./results/results_video"
out_path = results_dir + video_name.replace(".mp4", ".avi")
# crop_plates_dir = "D:/ALPR_Collections/ALPR_Faster_RCNN/crop_plates"

model = torch.load("LP_model_9616images_100e.pth")
model.eval()
# print(model)
device = torch.device("cuda")  # use GPU to train
model.to(device)

# Sử dụng torchvision.transforms để chuẩn bị ảnh
transform = transforms.Compose([
    transforms.ToTensor(),  # Chuyển đổi ảnh sang tensor
])


# Read video
cap = cv2.VideoCapture(vid_dir+video_name)
# Get the total frame count
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
codec = cv2.VideoWriter_fourcc(*'XVID')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# file_name = os.path.join(out_path, 'out_' + vid_dir.split('/')[-1] + )
# out = cv2.VideoWriter(file_name, codec, fps, (width, height))
out = cv2.VideoWriter(out_path, codec, fps, (width, height))

# Frame count variable.
ct = 0
# Reading video frame by frame.
while cap.isOpened():
    ret, image = cap.read()
    if ret:
        try:
            print(str(ct) + "/" + str(total_frames))
            start = time.time()

            # Chuyển đổi ảnh và thêm một chiều batch
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                prediction = model(image_tensor)
                print(prediction[0])

            # Chuyển tensor sang CPU và numpy array
            boxes = prediction[0]['boxes'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()

            # Chỉ lấy bounding box đầu tiên
            for box, score in zip(boxes, scores):
                if score > 0.8:
                    # Vẽ bounding box lên ảnh
                    x_min, y_min, x_max, y_max = map(int, box)
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(image, f'Score: {score:.4f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)
                    # Lưu plate đã detect vào thư mục crop_plates
                    # plate = image[y_min:y_max, x_min:x_max]  # Cắt và lưu phần plate đã detect
                    # plate_path = os.path.join(crop_plates_dir, file)
                    # cv2.imwrite(plate_path, plate)
            total_time = time.time() - start
            fps = round(1.0 / float(total_time), 2)
            cv2.putText(image, 'frame: %d fps: %s' % (ct, fps),
                        (0, int(100 * 1)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
            ct = ct + 1
            out.write(image)
        except Exception as e:
            print(e)
            continue
    else:
        break

# end = time.time()
# print("Average time:", (end-start)/float(ct))

