from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Tải hình ảnh
image_path = "anh.2.jpg"  # Đường dẫn đến ảnh của bạn
image = Image.open(image_path)

# Chuyển đổi hình ảnh thành mảng numpy để xử lý
image_array = np.array(image)

# 1. Ảnh âm bản
negative_image = 255 - image_array

# 2. Tăng độ tương phản bằng cách Kéo dãn độ tương phản (Chuẩn hóa)
min_val, max_val = np.min(image_array), np.max(image_array)
contrast_stretched_image = (image_array - min_val) * (255 / (max_val - min_val))

# 3. Biến đổi Logarit
c = 255 / np.log(1 + np.max(image_array))
log_transformed_image = c * np.log(1 + image_array)

# 4. Cân bằng histogram
def histogram_equalization(img):
    # Trải ảnh thành mảng 1 chiều
    img_flat = img.flatten()

    # Tính histogram
    hist, bins = np.histogram(img_flat, bins=256, range=[0, 256])

    # Hàm phân phối tích lũy (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]

    # Sử dụng nội suy tuyến tính để áp dụng CDF lên ảnh
    img_equalized = np.interp(img_flat, bins[:-1], cdf_normalized)

    # Đưa về hình dạng gốc của ảnh
    return img_equalized.reshape(img.shape)

# Áp dụng cân bằng histogram
hist_eq_image = histogram_equalization(image_array)

# Vẽ tất cả các kết quả
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].imshow(negative_image.astype(np.uint8))
axes[0, 0].set_title("Ảnh âm bản")
axes[0, 0].axis('off')

axes[0, 1].imshow(contrast_stretched_image.astype(np.uint8))
axes[0, 1].set_title("Ảnh Kéo dãn độ tương phản")
axes[0, 1].axis('off')

axes[1, 0].imshow(log_transformed_image.astype(np.uint8))
axes[1, 0].set_title("Ảnh Biến đổi Logarit")
axes[1, 0].axis('off')

axes[1, 1].imshow(hist_eq_image.astype(np.uint8))
axes[1, 1].set_title("Ảnh Cân bằng Histogram")
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()
