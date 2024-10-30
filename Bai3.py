import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Thiết lập đường dẫn đến thư mục chứa ảnh và file labels.csv
image_dir = r"D:\XLA-TGMT\Bai 3\image_paths"
label_path = r"D:\XLA-TGMT\Bai 3\labels\labels.csv"  # Đảm bảo đường dẫn chính xác

# Đọc dữ liệu nhãn
labels_data = pd.read_csv(label_path)
labels_map = {row["image_name"]: row["label"] for _, row in labels_data.iterrows()}

# Cài đặt các thông số hiển thị
img_dimensions = (200, 200)  # Kích thước ảnh để thu nhỏ và bố trí
padding = 20  # Khoảng cách giữa các ảnh
label_space = 30  # Không gian dành cho nhãn phía dưới mỗi ảnh

# Tạo danh sách tên tệp ảnh và xác định số lượng hàng, cột để bố trí
images_list = [img for img in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, img))]
total_images = len(images_list)
columns = 4
rows = (total_images + columns - 1) // columns

# Khởi tạo khung trắng cho bộ sưu tập ảnh và nhãn
total_height = (img_dimensions[1] + label_space + padding) * rows + padding
total_width = (img_dimensions[0] + padding) * columns + padding
gallery_image = np.full((total_height, total_width, 3), 255, dtype=np.uint8)  # Khung nền trắng

# Thêm từng ảnh và nhãn vào khung đã tạo
for i, img_name in enumerate(images_list):
    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        # Chỉnh sửa kích thước ảnh và thêm viền
        img = cv2.resize(img, img_dimensions)
        cv2.rectangle(img, (0, 0), (img_dimensions[0] - 1, img_dimensions[1] - 1), (0, 255, 0), 2)

        # Lấy nhãn từ labels_map
        img_label = labels_map.get(img_name, "unknown")

        # Tính toán vị trí hàng và cột để đặt ảnh
        current_row = i // columns
        current_col = i % columns
        y_start = current_row * (img_dimensions[1] + label_space + padding) + padding
        y_end = y_start + img_dimensions[1]
        x_start = current_col * (img_dimensions[0] + padding) + padding
        x_end = x_start + img_dimensions[0]

        # Đặt ảnh vào vị trí tương ứng trong khung
        gallery_image[y_start:y_end, x_start:x_end] = img

        # Thêm nhãn bên dưới mỗi ảnh
        cv2.putText(gallery_image, img_label, (x_start, y_end + label_space - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

# Hiển thị bộ sưu tập ảnh và nhãn sử dụng matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(gallery_image, cv2.COLOR_BGR2RGB))  # Chuyển đổi màu BGR sang RGB để hiển thị
plt.axis('off')  # Ẩn trục
plt.title("Image Gallery")
plt.show()
