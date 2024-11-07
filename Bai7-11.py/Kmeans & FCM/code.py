import cv2
import numpy as np
from sklearn.cluster import KMeans
import skfuzzy as fuzz  # Thư viện cho Fuzzy C-Means
import matplotlib.pyplot as plt

# Đường dẫn tới ảnh vệ tinh
image_path = 'anhdaura/anhvetinh1.png'  # Thay bằng đường dẫn tới ảnh của bạn

# Đọc ảnh vệ tinh và chuyển đổi sang ảnh xám (grayscale)
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Chuyển đổi ảnh thành dữ liệu dạng mảng để phân cụm
pixels = gray_image.reshape((-1, 1))

# Tiền xử lý: Chuẩn hóa dữ liệu về khoảng [0, 1]
pixels = np.float32(pixels) / 255.0

# Hàm thực hiện phân cụm với K-Means
def kmeans_clustering(pixels, K=3):
    kmeans = KMeans(n_clusters=K, random_state=0).fit(pixels)
    clustered = kmeans.labels_
    clustered_img = clustered.reshape(gray_image.shape)
    return clustered_img

# Hàm thực hiện phân cụm với Fuzzy C-Means (FCM)
def fcm_clustering(pixels, K=3):
    # Tạo dữ liệu đầu vào cho FCM
    pixels = pixels.T  # FCM yêu cầu dữ liệu dưới dạng (1, n_samples)
    
    # Áp dụng FCM
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        pixels, K, 2, error=0.005, maxiter=1000, init=None)
    
    # Lấy nhãn từ ma trận mức độ thành viên u
    labels = np.argmax(u, axis=0)
    clustered_img = labels.reshape(gray_image.shape)
    return clustered_img

# Thực hiện phân cụm K-Means và FCM
K = 3  # Số cụm cần phân chia (có thể điều chỉnh để phù hợp với dữ liệu)
kmeans_result = kmeans_clustering(pixels, K)
fcm_result = fcm_clustering(pixels, K)

# Hiển thị ảnh gốc và ảnh phân cụm
plt.figure(figsize=(12, 6))

# Hiển thị ảnh gốc
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Ảnh gốc")
plt.axis('off')

# Hiển thị kết quả K-Means
plt.subplot(1, 3, 2)
plt.imshow(kmeans_result, cmap='viridis')
plt.title("K-Means Clustering")
plt.axis('off')

# Hiển thị kết quả FCM
plt.subplot(1, 3, 3)
plt.imshow(fcm_result, cmap='viridis')
plt.title("FCM Clustering")
plt.axis('off')

plt.show()
