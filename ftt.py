import numpy as np
import matplotlib as plt


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


def fft_2d():
    return None

def inverse_fft_2d():
    return None

def main():
    x = np.random.rand(8)

    X = fft(x)
    x_reconstructed = inverse_fft(X)


    print("original:", x)
    print("reconstructed:", np.real_if_close(x_reconstructed))
    print("close?", np.allclose(x, x_reconstructed))


if __name__ == "__main__":
    main()











    



