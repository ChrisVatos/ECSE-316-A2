import numpy as np
from fft import dft, fft, inverse_fft, fft_2d, dft_2d, inverse_fft_2d

def test_custom_dft():
    """
    Test the custom naive implementation of DFT against numpy's FFT function
    """
    # Generate 1D array of random numbers
    test_arr = np.random.rand(10)

    # Compute 1D DFT using custom naive DFT implementation
    custom_dft = dft(test_arr)

    # Compute 1D DFT using numpy's FFT
    numpy_fft = np.fft.fft(test_arr)

    assert np.allclose(custom_dft, numpy_fft), "Custom DFT output does not match numpy's FFT output"


def test_custom_fft():
    """
    Test the custom implementation of FFT against numpy's FFT function
    """
    # Generate 1D array of random numbers (Size of array needs to be a power of 2)
    test_arr = np.random.rand(16) 

    # Compute 1D FFT using custom FFT implementation
    custom_fft = fft(test_arr)

    # Compute 1D FFT using numpy's FFT
    numpy_fft = np.fft.fft(test_arr)

    assert np.allclose(custom_fft, numpy_fft), "Custom FFT output does not match numpy's FFT output"


def test_custom_inverse_fft():
    """
    Test the custom implementation of inverse FFT against numpy's inverse FFT function
    """
    # Generate 1D array of random numbers (Size of array needs to be a power of 2)
    test_arr = np.random.rand(16) 

    # Compute 1D FFT using numpy's FFT
    numpy_fft = np.fft.fft(test_arr)

    # Compute inverse FFT using custom inverse FFT implementation
    custom_inverse_fft = inverse_fft(numpy_fft)

    # Compute inverse FFT using numpy's inverse FFT
    numpy_inverse_fft = np.fft.ifft(numpy_fft)

    assert np.allclose(custom_inverse_fft, numpy_inverse_fft), "Custom inverse FFT output does not match numpy's inverse FFT output"
    assert np.allclose(custom_inverse_fft, test_arr), "Custom inverse FFT did not recover the original array"

def test_custom_2d_fft():
    """
    Test the custom implementation of 2D FFT against numpy's 2D FFT function
    """
    # Generate 2D array of random numbers (Both the length and width of array need to be powers of 2)
    test_arr = np.random.rand(16, 16)

    # Compute 2D FFT using custom 2D FFT implementation
    custom_2d_fft = fft_2d(test_arr)

    # Compute 2D FFT using numpy's 2D FFT
    numpy_2d_fft = np.fft.fft2(test_arr)

    assert np.allclose(custom_2d_fft, numpy_2d_fft), "Custom 2D FFT output does not match numpy's 2D FFT output"


def test_custom_2d_dft():
    """
    Test the custom naive implementation of 2D DFT against numpy's 2D FFT function
    """
    # Generate 2D array of random numbers
    test_arr = np.random.rand(10, 10)

    # Compute 2D DFT using custom naive 2D DFT implementation
    custom_2d_dft = dft_2d(test_arr)

    # Compute 2D DFT using numpy's 2D FFT
    numpy_2d_fft = np.fft.fft2(test_arr)

    assert np.allclose(custom_2d_dft, numpy_2d_fft), "Custom 2D DFT output does not match numpy's 2D FFT output"


def test_custom_inverse_2d_fft():
    """
    Test the custom implementation of inverse 2D FFT against numpy's inverse 2D FFT function
    """
    # Generate 2D array of random numbers (Both the length and width of array need to be powers of 2)
    test_arr = np.random.rand(16, 16)

    # Compute 2D FFT using numpy's 2D FFT
    numpy_2d_fft = np.fft.fft2(test_arr)

    # Compute inverse 2D FFT using custom inverse 2D FFT implementation
    custom_inverse_2d_fft = inverse_fft_2d(numpy_2d_fft)

    # Compute inverse 2D FFT using numpy's inverse 2D FFT
    numpy_inverse_2d_fft = np.fft.ifft2(numpy_2d_fft)

    assert np.allclose(custom_inverse_2d_fft, numpy_inverse_2d_fft), "Custom inverse 2D FFT output does not match numpy's inverse 2D FFT output"
    assert np.allclose(custom_inverse_2d_fft, test_arr), "Custom inverse 2D FFT did not recover the original array"