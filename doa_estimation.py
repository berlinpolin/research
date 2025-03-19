import numpy as np
from signal_model import manifold_matrix

def main():
    num_sensor = 4
    num_sub_sensor = num_sensor - 1
    d = 0.5 * np.arange(num_sensor) # half-wavelenght spacing
    doa = np.array([30, 60])
    num_path = len(doa)
    s = np.eye(num_path)
    h = manifold_matrix(1, d, doa, num_sensor)
    x = h@s
    corr_x = x@(x.conj().T)
    u, _, _ = np.linalg.svd(corr_x)
    j1 = np.concatenate((np.eye(num_sub_sensor), np.zeros([num_sub_sensor, 1])),\
                         axis=1)
    j2 = np.concatenate((np.zeros([num_sub_sensor, 1]), np.eye(num_sub_sensor)),\
                         axis=1)
    q, r = np.linalg.qr(j1@u[:, :num_path])
    sro = np.linalg.solve(r, q.conj().T @ j2 @ u[:, :num_path])



if __name__ == "__main__":
    main()