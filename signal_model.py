import numpy as np

def steering_vec(
    wavelength,  # m
    r: np.ndarray,  # (num_element,) distance, m
    theta,  # degree
    num_element,
):
    a = np.zeros(num_element, dtype=np.complex_)
    for n in range(num_element):
        a[n] = np.exp(-2j * np.pi * r[n] * np.sin(theta * np.pi / 180) / wavelength)

    return a

def manifold_matrix(
    wavelength,  # m
    r: np.ndarray,  # (num_element,) distance, m
    theta: np.ndarray,  # (num_angle,) degree
    num_element,
):
    num_theta = len(theta)
    s = np.zeros((num_element, num_theta), dtype=np.complex_)
    for n in range(num_theta):
        s[:, n] = steering_vec(wavelength, r, theta[n], num_element)

    return s