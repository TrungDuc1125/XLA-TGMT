import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đường dẫn đến ảnh (sử dụng dấu \\ để tránh xung đột ký tự đặc biệt)
image_path = r'anh.2.jpg'

# Đọc ảnh gốc dưới dạng ảnh xám
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Cannot read the image from the given path. Please check the file path.")
else:
    # 1. Dò biên với toán tử Sobel
    sobel_x = cv2.filter2D(image, cv2.CV_64F, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))  # Gradient theo trục x
    sobel_y = cv2.filter2D(image, cv2.CV_64F, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))  # Gradient theo trục y
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)  # Tổng hợp gradient theo cả hai hướng

    # 2. Dò biên với toán tử Laplace Gaussian
    log_kernel = np.array([[0, 0, -1, 0, 0],
                           [0, -1, -2, -1, 0],
                           [-1, -2, 16, -2, -1],
                           [0, -1, -2, -1, 0],
                           [0, 0, -1, 0, 0]])

    log_image = cv2.filter2D(image, -1, log_kernel)

    # Hiển thị các kết quả
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image (Grayscale)')
    plt.axis('off')  # Ẩn trục tọa độ
    
    plt.subplot(1, 3, 2)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Edge Detection with Sobel')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(log_image, cmap='gray')
    plt.title('Edge Detection with Laplacian Gaussian')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
