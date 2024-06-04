import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk meresize gambar menjadi ukuran persegi
def resize_to_square(image, size=256):
    return cv2.resize(image, (size, size))

# Fungsi untuk mengubah gambar menjadi grayscale
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Fungsi untuk menerapkan high-pass filter (HPF)
def apply_hpf(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Membuat mask dengan 0 di tengah (low-pass area) dan 1 di luar (high-pass area)
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 30
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = np.sqrt((x - center[0])**2 + (y - center[1])**2) <= r
    mask[mask_area] = 0

    # Terapkan mask dan inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back

# Fungsi untuk menerapkan Fast Fourier Transform (FFT)
def apply_fft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum

# Fungsi untuk menentukan titik pola thenar dan hypothenar
def detect_thenar_hypothenar_points(image):
    # Asumsi: Titik thenar dan hypothenar adalah titik kontras tinggi dalam region tertentu
    height, width = image.shape
    thenar_region = image[0:height // 2, 0:width // 2]
    hypothenar_region = image[height // 2:height, width // 2:width]

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(thenar_region)
    thenar_point = max_loc
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hypothenar_region)
    hypothenar_point = (max_loc[0] + width // 2, max_loc[1] + height // 2)

    return thenar_point, hypothenar_point

# Main program
def main(image_path):
    # Baca gambar
    image = cv2.imread(image_path)

    # Resize gambar menjadi ukuran persegi
    image_square = resize_to_square(image)

    # Convert gambar ke grayscale
    grayscale_image = convert_to_grayscale(image_square)

    # Terapkan high-pass filter (HPF)
    hpf_image = apply_hpf(grayscale_image)

    # Terapkan Fast Fourier Transform (FFT)
    fft_image = apply_fft(grayscale_image)

    # Tentukan titik pola thenar dan hypothenar
    thenar_point, hypothenar_point = detect_thenar_hypothenar_points(grayscale_image)

    # Tampilkan hasil
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image_square, cv2.COLOR_BGR2RGB))
    plt.title('Resized Image')

    plt.subplot(2, 2, 2)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title('Grayscale Image')

    plt.subplot(2, 2, 3)
    plt.imshow(hpf_image, cmap='gray')
    plt.title('HPF Image')

    plt.subplot(2, 2, 4)
    plt.imshow(fft_image, cmap='gray')
    plt.title('FFT Image')

    plt.scatter([thenar_point[0], hypothenar_point[0]], [thenar_point[1], hypothenar_point[1]], color='red')
    plt.text(thenar_point[0], thenar_point[1], 'Thenar', color='red', fontsize=12)
    plt.text(hypothenar_point[0], hypothenar_point[1], 'Hypothenar', color='red', fontsize=12)

    plt.show()

# Jalankan program utama dengan path gambar palmprint
image_path = 'tangan_8.jpg'  # Ganti dengan path gambar palmprint Anda
main(image_path)
