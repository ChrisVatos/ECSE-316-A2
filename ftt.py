import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import os
from matplotlib.colors import LogNorm

# -----------------------------------------------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------------------------------------------#
# -------------------------------- Fourier and Inverse Fourier Transformation Functions -------------------------------- #
# -----------------------------------------------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------------------------------------------#
def dft(arr: np.ndarray) -> np.ndarray:
    N = arr.size
    n = np.arange(N)
    k = np.reshape(n, (N, 1))

    e = np.exp(-2j * np.pi * k * n / N)

    X_k = np.dot(e, arr)
    return X_k

def fft(arr: np.ndarray) -> np.ndarray:
    N = arr.size

    if N <= 1:
        return arr
    
    even_indices = fft(arr[0::2])
    odd_indices = fft(arr[1::2])

    result = np.zeros(N, dtype=complex)
    for k in range(N // 2):
        e = np.exp((-2j * np.pi * k) / N)
        result[k] = even_indices[k] + e * odd_indices[k]
        result[k + N // 2] = even_indices[k] - e * odd_indices[k]
    return result

def inverse_fft(arr: np.ndarray) -> np.ndarray:
    # Define an inner recursive function for inverse FFT to 
    # handle the divide and conquer approach
    def inverse_fft_recursive(arr: np.ndarray) -> np.ndarray:
        N = arr.size

        # Base Case
        if N <= 1:
            return arr
        
        even_indices = inverse_fft_recursive(arr[0::2])
        odd_indices = inverse_fft_recursive(arr[1::2])

        result = np.zeros(N, dtype=complex)
        for k in range(N // 2):
            e = np.exp((2j * np.pi * k) / N)
            result[k] = even_indices[k] + e * odd_indices[k]
            result[k + N // 2] = even_indices[k] - e * odd_indices[k]
        return result

    # Want to normalize the result returned by inverse_fft_recursive
    N = arr.size
    return  inverse_fft_recursive(arr) / N

def fft_2d(img: np.ndarray) -> np.ndarray:
    # As per the assignment description:
    # "Note that the term inside the brackets is a 1D-DFT of the rows of the 2D matrix of values f and
    # that the outer sum is another 1D-DFT over the transformed rows performed along each column"

    # 1D-DFT (FFT) for each row of the 2D image 
    transformed_rows = np.transpose([fft(row) for row in img]) 

    # 1D-DFT (FFT) along each column of the transformed rows
    freqs = np.transpose([fft(col) for col in transformed_rows])
    return freqs

def inverse_fft_2d(freqs: np.ndarray) -> np.ndarray:
    # Similar approach to fft_2d() but using inverse_fft() instead

    # 1D-IDFT (IFFT) for each row of the frequency domain  
    transformed_rows = np.transpose([inverse_fft(row) for row in freqs])

    # 1D-IDFT (IFFT) along each column of the transformed rows
    img = np.transpose([inverse_fft(row) for row in transformed_rows])
    return img

def is_power_of_2(n: int) -> bool:
    # A number n is a power of 2 iff it is positive and only a single bit 
    # is set in its binary representation. We can easily check this with a bitwise AND operations
    return (n > 0) and (n & (n - 1) == 0)

def load_img(img_path: str) -> np.ndarray:
    # Read the image using cv2
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # As per the assignment requirements, we need to ensure that the image dimensions are powers of 2
    height, width = img.shape
    
    # If the current height is not a power of 2, acquire the next power of 2
    if not is_power_of_2(height):
        height = 2 ** (height - 1).bit_length() 

    # If the current width is not a power of 2, acquire the next power of 2
    if not is_power_of_2(width):
        width = 2 ** (width - 1).bit_length() 

    # Resize the image to the new dimensions which are powers of 2    
    resized_img = cv2.resize(img, (width, height))
    return np.array(resized_img)

# -----------------------------------------------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------------------------------------------#
# --------------------------------------------- Functions for Modes 1 to 4 --------------------------------------------- #
# -----------------------------------------------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------------------------------------------#

def mode_1_fft_img(img: np.ndarray) -> np.ndarray:
    # Obtain the 2D FFT of the image
    fast_fourier_transformed_img = fft_2d(img)

    # Output a one by two subplot of the original image and next to it its Fourier transform
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.title("Original Image", fontweight="bold")
    plt.imshow(img, cmap='gray')

    # Plot 2D Fast Fourier Transformed image
    plt.subplot(1, 2, 2)
    plt.title("Fast Fourier Transform (FFT) of Image (Log Scaled)", fontweight="bold")
    plt.imshow(np.abs(fast_fourier_transformed_img), cmap='gray', norm=LogNorm())

    plt.show()

def mode_2_denoise_img():
    return None

def compress_img():
    return None

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="Mode to run program in", type=int, action="store", default=1)
    parser.add_argument("-i", help="Filename of the image we wish to take the DFT of", type=str, action="store", default="./assets/moonlanding.png")
    args = parser.parse_args()
    return args

def main():
    # Parse the command line arguments m (mode) and i (image)
    args = parse_command_line_args()

    # Ensure input image exists before calling corresponding functions based on mode
    # Check if the provided image path exsists; display error message if it does not
    # or if there was an error loading the image
    source_img = None
    try:
        if not os.path.exists(args.i):
            print(f"Error: The specified image path '{args.i}' does not exist.")
            return
        # Load the image from the specified path if the provided is valid and exists
        else:
            source_img = load_img(args.i)
    except Exception as e:
        print(f"An error occurred while loading the image: {e}")
        return

    # Calling the correct functions based on input parameters after verifying image path
    if args.m == 1:
        mode_1_fft_img(source_img)
    elif args.m == 2:
        return None
    elif args.m == 3:
        return None
    elif args.m == 4:
        return None
    else :
        print("Invalid mode selected. Please choose a mode between 1 and 4.")
        return None
    




if __name__ == "__main__":
    main()











    



