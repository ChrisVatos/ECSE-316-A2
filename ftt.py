import numpy as np
import matplotlib as plt
import cv2


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
    # that performed the divide and conquer approach
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

def denoising_img():
    return None

def compress_img():
    return None

def load_img():
    return None

def main():
    return None


if __name__ == "__main__":
    main()











    



