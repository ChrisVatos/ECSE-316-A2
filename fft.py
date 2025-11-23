import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import os
from matplotlib.colors import LogNorm
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------------------------------------------#
# -------------------------------- Fourier and Inverse Fourier Transformation Functions -------------------------------- #
# -----------------------------------------------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------------------------------------------#
def dft(arr: np.ndarray) -> np.ndarray:
    """
    Naive DFT algorithm implementation
    """
    N = arr.size
    n = np.arange(N)
    k = np.reshape(n, (N, 1))

    e = np.exp((-2j * np.pi * k * n) / N)

    X_k = np.dot(e, arr)
    return X_k

def fft(arr: np.ndarray) -> np.ndarray:
    """
    Cooley-Tukey FFT algorithm implementation
    """
    N = arr.size

    # Base Case
    if N <= 1:
        return arr
    
    # Split the sum in the even and odd indices
    even_indices = fft(arr[0::2])
    odd_indices = fft(arr[1::2])

    # Initialize the result array
    result = np.zeros(N, dtype=complex)
    for k in range(N // 2):
        e = np.exp((-2j * np.pi * k) / N)
        result[k] = even_indices[k] + e * odd_indices[k]
        result[k + N // 2] = even_indices[k] - e * odd_indices[k]
    return result

def inverse_fft(arr: np.ndarray) -> np.ndarray:
    """
    Inverse Cooley-Tukey FFT algorithm implementation
    """
    # Defined an inner recursive function for inverse FFT to 
    # return the result prior to scaling by 1/N
    def inverse_fft_recursive(arr: np.ndarray) -> np.ndarray:
        N = arr.size

        # Base Case
        if N <= 1:
            return arr
        
        # Split the sum in the even and odd indices
        even_indices = inverse_fft_recursive(arr[0::2])
        odd_indices = inverse_fft_recursive(arr[1::2])

        # Initialize the result array
        result = np.zeros(N, dtype=complex)
        for k in range(N // 2):
            e = np.exp((2j * np.pi * k) / N)
            result[k] = even_indices[k] + e * odd_indices[k]
            result[k + N // 2] = even_indices[k] - e * odd_indices[k]
        return result

    # Want to normalize/scale the result returned by 
    # inverse_fft_recursive by 1/N
    N = arr.size
    return  inverse_fft_recursive(arr) / N

def fft_2d(img: np.ndarray) -> np.ndarray:
    """
    2D FFT implementation using the Cooley-Tukey algorithm
    """
    # As per the assignment description:
    # "Note that the term inside the brackets is a 1D-DFT of the rows of the 2D matrix of values f and
    # that the outer sum is another 1D-DFT over the transformed rows performed along each column"

    # FFT for each row of the image & transpose the result
    transformed_rows = np.transpose([fft(row) for row in img]) 

    # FFT along each column of the transformed + transposed rows
    # Taking the 2D FFT returns the frequency domain representation of the provided image
    freqs = np.transpose([fft(col) for col in transformed_rows])
    return freqs

def dft_2d(img: np.ndarray) -> np.ndarray:
    """
    2D DFT implementation using the naive DFT algorithm
    """
    # As per the assignment description:
    # "Note that the term inside the brackets is a 1D-DFT of the rows of the 2D matrix of values f and
    # that the outer sum is another 1D-DFT over the transformed rows performed along each column"

    # DFT for each row of the image & transpose the result
    transformed_rows = np.transpose([dft(row) for row in img])

    # DFT along each column of the transformed + transposed rows
    # Taking the 2D DFT returns the frequency domain representation of the provided image
    freqs = np.transpose([dft(col) for col in transformed_rows])
    return freqs

def inverse_fft_2d(freqs: np.ndarray) -> np.ndarray:
    """
    2D Inverse FFT implementation using the Cooley-Tukey algorithm
    """
    # Similar approach to fft_2d() but using inverse_fft() instead of fft()

    # IFFT for each row of the frequency domain & transpose the result
    transformed_rows = np.transpose([inverse_fft(row) for row in freqs])

    # IFFT along each column of the transformed + transposed rows
    # Taking the 2D inverse FFT returns the original image
    img = np.transpose([inverse_fft(row) for row in transformed_rows])
    return img

def is_power_of_2(n: int) -> bool:
    """
    Checks if a number is a power of 2
    """
    # A number n is a power of 2 iff it is positive and only a single bit is set 
    # in its binary representation. We can easily check this with a bitwise AND operation
    return (n > 0) and (n & (n - 1) == 0)

def load_img(img_path: str) -> np.ndarray:
    """
    Loads an image in grayscale from the specified path and resizes it to have
    dimensions that are powers of 2 (if necessary).
    """
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
    """
    Computes and displays the FFT of the provided image
    """
    # Compute the 2D FFT of the image
    fast_fourier_transformed_img = fft_2d(img)

    # Output a one by two subplot of the original image and next to it its Fourier transform
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')

    # Plot 2D Fast Fourier Transformed image
    plt.subplot(1, 2, 2)
    plt.title("FFT of Original Image (Log Scaled)")
    plt.imshow(np.abs(fast_fourier_transformed_img), cmap='gray', norm=LogNorm())

    plt.show()

def mode_2_denoise_img(img: np.ndarray, high_frec_percent_remove: float = 0.95) -> np.ndarray:
    """
    Denoises and displays the image by removing high frequency components in the frequency domain
    """
    # Code Reference: https://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_fft_image_denoise.html
    # Compute 2D FFT of the image
    fast_fourier_transformed_img = fft_2d(img)

    # After acquiring the 2D FFT, the low frequency components are located at the four corners 
    # of the fourier transform's frequency domain while the high frequency components are located in the center. We will 
    # therefore zero out the set the pecified percentage of the high frequency components in the center, thus keeping only
    # the low frequency components located at the corners of the frequency domain.
    num_rows, num_cols = fast_fourier_transformed_img.shape

    # Determine the number of rows and columns to remove based on the specified percentage (high_frec_percent_remove)
    num_rows_remove = int(num_rows * high_frec_percent_remove)
    num_cols_remove = int(num_cols * high_frec_percent_remove)

    # Based on the total number of rows and columns to remove, determine the start and end indices for each 
    start_index_row_to_remove = int((num_rows - num_rows_remove) / 2)
    end_index_row_to_remove = int(start_index_row_to_remove + num_rows_remove)
    start_index_col_to_remove = int((num_cols - num_cols_remove) / 2)
    end_index_col_to_remove = int(start_index_col_to_remove + num_cols_remove)

    # We set all rows with indices between start_index_row_to_remove and end_index_row_to_remove to 0
    # in order to remove a specified percentage (high_frec_percent_remove) of the high frequency components
    fast_fourier_transformed_img[start_index_row_to_remove:end_index_row_to_remove] = 0

    # We set all cols with indices between start_index_col_to_remove and end_index_col_to_remove to 0
    # in order to remove a specified percentage (high_frec_percent_remove) of the high frequency components
    fast_fourier_transformed_img[:, start_index_col_to_remove:end_index_col_to_remove] = 0

    # Compute the inverse 2D FFT to get back the filtered original image
    denoised_img = inverse_fft_2d(fast_fourier_transformed_img)

    # Count and display the number of non-zero frequency components remaining after denoising
    count_non_zero = np.count_nonzero(fast_fourier_transformed_img)
    num_coefs = fast_fourier_transformed_img.size
    ratio_non_zero = count_non_zero / num_coefs # This also represents the ratio of low frequency components kept after denoising
    print(f"Number of non-zero frequency components after denoising: {count_non_zero}")
    print(f"Ratio of non-zero frequency components after denoising: {ratio_non_zero:.2%}")

    # Output a one by two subplot of the original image and next to it its denoised version
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')

    # Plot denoised image
    plt.subplot(1, 2, 2)
    plt.title(f"Denoised Image: Removed {high_frec_percent_remove:.1%} of High Frequencies")
    plt.imshow(np.abs(denoised_img), cmap='gray')
    plt.show()

def mode_3_compress_img(img: np.ndarray):
    thresholds = [0, 50, 75, 90, 99, 99.9]  # Percentiles for compression levels
    # take the FFT of the image to compress.
    fast_fourier_transformed_img = fft_2d(img)


    # compression comes from setting some Fourier coefficients to zero
    # 1. you can threshold the coefficients magnitude and take only the largest percentile of them
    for threshold in thresholds:
        #determine the magnitude threshold for the current percentile
        magnitude_threshold = np.percentile(np.abs(fast_fourier_transformed_img), threshold)

        compressed_coeffs = np.copy(fast_fourier_transformed_img)

        #set coefficients with magnitude below the threshold to zero
        compressed_coeffs[np.abs(compressed_coeffs) < magnitude_threshold] = 0

        compressed_img = inverse_fft_2d(compressed_coeffs)

        #count and print the number of nonzero coefficients
        count_non_zero = np.count_nonzero(compressed_coeffs)
        print(f"Threshold: {threshold}%, Non-zero coefficients: {count_non_zero}")

        sparse_matrix_path = f"assets/compressed_coeffs_{threshold}percent.npz"
        np.savez_compressed(sparse_matrix_path, compressed_coeffs=compressed_coeffs)
        sparse_matrix_size = os.path.getsize(sparse_matrix_path) / 1024  # Size in KB
        print(f"Threshold: {threshold}%, Sparse matrix size: {sparse_matrix_size:.2f} KB")

        # Display the compressed image
        plt.subplot(2, 3, thresholds.index(threshold) + 1)
        plt.title(f"Compression: {threshold}%")
        plt.imshow(np.abs(compressed_img), cmap='gray')

        #save each figure individually
        plt.figure()  
        plt.title(f"Compression: {threshold}%")
        plt.imshow(np.abs(compressed_img), cmap='gray')
        plt.savefig(f"assets/compressed_image_{threshold}percent.png", dpi=300)  # Save the figure
        plt.close() 

    plt.show()

   

    return None

def mode_4_plot_runtimes():
    
    sizes = [2**i for i in range(5, 11)]  
    num_trials = 10  
    confidence_interval = 2  # 97% confidence interval (2 * std deviation)

    dft_means = []
    dft_stds = []
    fft_means = []
    fft_stds = []

    for size in sizes:
        dft_runtimes = []
        fft_runtimes = []

        for _ in range(num_trials):
            print(f"Running trial for size {size}x{size}...")
            random_array = np.random.rand(size, size)

            start_time = time.time()
            dft_2d(random_array)
            dft_runtimes.append(time.time() - start_time)

            start_time = time.time()
            fft_2d(random_array)
            fft_runtimes.append(time.time() - start_time)

        dft_means.append(np.mean(dft_runtimes))
        dft_stds.append(np.std(dft_runtimes))

        fft_means.append(np.mean(fft_runtimes))
        fft_stds.append(np.std(fft_runtimes))

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.errorbar(sizes, dft_means, yerr=[confidence_interval * std for std in dft_stds],
                 label="Naive DFT", fmt='-o', capsize=5)
    plt.errorbar(sizes, fft_means, yerr=[confidence_interval * std for std in fft_stds],
                 label="FFT", fmt='-o', capsize=5)

    plt.xlabel("Problem Size (N x N)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Comparison: Naive DFT vs FFT")
    plt.xscale("log", base=2)  # Log scale for x-axis (powers of 2)
    plt.yscale("log")  # Log scale for y-axis
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig("assets/runtime_comparison.png", dpi=300)  # Save the plot as a high-quality PNG
    plt.show()

    print("Runtime comparison plot saved as 'runtime_comparison.png'.")


    return None

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", 
        help="Mode to run program in: 1 = FFT Display, 2 = Denoising, 3 = Compression, 4 = Plotting Runtimes", 
        type=int, 
        action="store", 
        default=1)
    parser.add_argument(
        "-i", 
        help="Filename of the image we wish to take the DFT of. Default is ./assets/moonlanding.png", 
        type=str, 
        action="store", 
        default="./assets/moonlanding.png")
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
        mode_2_denoise_img(source_img, 0.925)
    elif args.m == 3:
        mode_3_compress_img(source_img)
    elif args.m == 4:
        mode_4_plot_runtimes()
    else :
        print("Invalid mode selected. Please choose a mode between 1 and 4.")
        return None
    

if __name__ == "__main__":
    main()











    



