import numpy as np
from numba import cuda, float64

@cuda.jit
def compute_gram_elements_kernel(A, g_diag, g_offdiag):
    """
    Kiszámítja az oszlopok normáit és a páronkénti skaláris szorzatokat (Gram-mátrix elemek).
    """
    col_i, col_j = cuda.grid(2)
    n = A.shape[1]
    
    if col_i < n and col_j < n and col_i < col_j:
        m = A.shape[0]
        dot_ii = 0.0
        dot_jj = 0.0
        dot_ij = 0.0
        
        for k in range(m):
            aik = A[k, col_i]
            ajk = A[k, col_j]
            dot_ii += aik * aik
            dot_jj += ajk * ajk
            dot_ij += aik * ajk
            
        g_diag[col_i] = dot_ii
        g_diag[col_j] = dot_jj
        g_offdiag[col_i, col_j] = dot_ij

@cuda.jit
def jacobi_rotate_kernel(A, V, col_i, col_j, c, s):
    """
    Givens-forgatás alkalmazása az A mátrix oszlopaira és a V mátrixra.
    """
    m = A.shape[0]
    row = cuda.grid(1)
    
    if row < m:
        a_ki = A[row, col_i]
        a_kj = A[row, col_j]
        A[row, col_i] = a_ki * c - a_kj * s
        A[row, col_j] = a_ki * s + a_kj * c
        
    n = V.shape[1]
    if row < n:
        v_ki = V[row, col_i]
        v_kj = V[row, col_j]
        V[row, col_i] = v_ki * c - v_kj * s
        V[row, col_j] = v_ki * s + v_kj * c


def one_sided_jacobi_svd_gpu(A_host, max_sweeps=50, tol=1.0e-9):
    """
    Host oldali vezérlő a One-Sided Jacobi SVD iterációhoz Numba CUDA-val.
    """
    m, n = A_host.shape
    A_dev = cuda.to_device(A_host.astype(np.float64))
    V_host = np.eye(n, dtype=np.float64)
    V_dev = cuda.to_device(V_host)
    
    g_diag_dev = cuda.device_array(n, dtype=np.float64)
    g_offdiag_dev = cuda.device_array((n, n), dtype=np.float64)
    
    threadsperblock = (16, 16)
    blockspergrid_x = (n + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (n + threadsperblock[1] - 1) // threadsperblock[1]
    grid_dim = (blockspergrid_x, blockspergrid_y)
    
    block_1d = 256
    grid_1d = (max(m, n) + block_1d - 1) // block_1d

    for sweep in range(max_sweeps):
        compute_gram_elements_kernel[grid_dim, threadsperblock](A_dev, g_diag_dev, g_offdiag_dev)
        g_offdiag_host = g_offdiag_dev.copy_to_host()
        
        max_offdiag = 0.0
        p_idx, q_idx = -1, -1
        for i in range(n):
            for j in range(i + 1, n):
                val = abs(g_offdiag_host[i, j])
                if val > max_offdiag:
                    max_offdiag = val
                    p_idx, q_idx = i, j
                    
        if max_offdiag < tol:
            break
            
        # Host oldali számítás a 2x2-es forgatási szögre (Givens parameters c, s)
        # Az átláthatóság kedvéért itt letöltjük a szükséges elemeket
        col_p_host = A_dev[:, p_idx].copy_to_host()
        col_q_host = A_dev[:, q_idx].copy_to_host()
        
        alpha = np.dot(col_p_host, col_p_host)
        beta = np.dot(col_q_host, col_q_host)
        gamma = np.dot(col_p_host, col_q_host)
        
        if abs(gamma) < tol * np.sqrt(alpha * beta):
            continue
            
        zeta = (beta - alpha) / (2.0 * gamma)
        t = np.sign(zeta) / (abs(zeta) + np.sqrt(1.0 + zeta * zeta))
        c = 1.0 / np.sqrt(1.0 + t * t)
        s = t * c
        
        # Forgatás alkalmazása GPU-n
        jacobi_rotate_kernel[grid_1d, block_1d](A_dev, V_dev, p_idx, q_idx, c, s)

    # Eredmények kinyerése
    A_final = A_dev.copy_to_host()
    V_final = V_dev.copy_to_host()
    
    # Szinguláris értékek kiszámítása az oszlopok normáiból
    singular_values = np.linalg.norm(A_final, axis=0)
    
    # U mátrix előállítása (oszlopok normalizálása)
    U_final = np.zeros_like(A_final)
    for i in range(n):
        if singular_values[i] > tol:
            U_final[:, i] = A_final[:, i] / singular_values[i]
            
    # Rendezés csökkenő sorrendbe
    sorted_indices = np.argsort(singular_values)[::-1]
    singular_values = singular_values[sorted_indices]
    U_final = U_final[:, sorted_indices]
    V_final = V_final[:, sorted_indices]
    
    return U_final, singular_values, V_final.T


if __name__ == "__main__":
    # Tesztmátrix generálás
    np.random.seed(42)
    A_test = np.random.randn(64, 32)
    
    print("GPU SVD futtatása...")
    U_gpu, S_gpu, Vt_gpu = one_sided_jacobi_svd_gpu(A_test)
    
    # Referencia NumPy SVD összehasonlításhoz
    U_ref, S_ref, Vt_ref = np.linalg.numpy.linalg.svd(A_test, full_matrices=False) if hasattr(np.linalg, 'numpy') else np.linalg.svd(A_test, full_matrices=False)
    
    print("\n--- Numerikus Korrektség Ellenőrzés ---")
    print(f"Legnagyobb szinguláris érték eltérés: {np.max(np.abs(S_gpu - S_ref))}")
    reconstruction_error = np.linalg.norm(A_test - np.dot(U_gpu * S_gpu, Vt_gpu)) / np.linalg.norm(A_test)
    print(f"Relatív rekonstrukciós hiba (||A - USV^T|| / ||A||): {reconstruction_error:.2e}")