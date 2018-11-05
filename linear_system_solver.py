# Linear System Solvers
#
# Available Methods:
# - LU (Own Implementation)
# - LU_sparse (Own Implementation)
# - LU_compiled (Own Implementation)
# - LU_scipy_sparse (Scipy Implementation)
# - QR (Own Implementation)
# - QR_compiled (Own Implementation)
# - QR_scipy (Scipy Implementation)
# - scipy (Scipy Implementation)
# - jacobi (Own Implementation)
# - jacobi_compiled (Own Implementation)
# - jacobi_sparse (Own Implementation)
# - gauss_seidel (Own Implementation)
# - gauss_seidel_compiled (Own Implementation)
# - gauss_seidel_sparse (Own Implementation)
# - sor (Own Implementation)
# - sor_compiled (Own Implementation)
# - sor_sparse (Own Implementation)
#
# Assignment 1 for ME4233
# Author: Giovani Hidalgo Ceotto
# Prof: Mengqi Zhang

import numpy as np
from scipy.sparse import dia_matrix, csr_matrix, csc_matrix, lil_matrix, identity, tril, triu
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import inv as sparse_inv
from scipy.linalg import solve, inv, qr
from numba import jit

# Wrapper for all solvers
def solve_algebraic_system(A, b, method, initial_guess=0, true_solution=0, tol=1e-6, w=1):
    """ Solves the algebraic system A*x = b, for the vector x.
        
        Parameters
        ----------
        A : array
            Matrix A corresponding to the system A*x = b. Must be in sparse
            format.
        b : array
            Column vector b corresponding to the system A*x = b.
        method : string
            Mehtod to be used. Available methods are: 
                - LU (Own Implementation)
                - LU_sparse (Own Implementation)
                - LU_compiled (Own Implementation)
                - LU_scipy_sparse (Scipy Implementation)
                - QR (Own Implementation)
                - QR_compiled (Own Implementation)
                - QR_scipy (Scipy Implementation)
                - scipy (Scipy Implementation)
                - jacobi (Own Implementation)
                - jacobi_compiled (Own Implementation)
                - jacobi_sparse (Own Implementation)
                - gauss_seidel (Own Implementation)
                - gauss_seidel_compiled (Own Implementation)
                - gauss_seidel_sparse (Own Implementation)
                - sor (Own Implementation)
                - sor_compiled (Own Implementation)
                - sor_sparse (Own Implementation)
        initial_guess : array, optional
            Initial solution used for iterative solvers.
        true_solution : array, optional
            If given, iterative solver residuals will be calculated considering
            this.
        tol : float, optional
            Convergence tolerance for iterative solvers.
        w : float, optiona
            Relaxation parameter to be used with the SOR methods.

        Returns
        -------
        x : array
            Solution vector corresponding to A*x = b.
        residue : array
            Residue array given by iterative solver.        
    """
    if method == "LU":
        L_prime, U = LU_factorization(A.toarray())
        return solve_upper_triangular_system(U, L_prime.dot(b))
    elif method == "LU_sparse":
        L_prime, U = LU_factorization_sparse(A)
        return solve_upper_triangular_system(U.toarray(), L_prime.dot(b))
    elif method == "LU_compiled":
        L, U = LU_factorization_compiled(A.toarray())
        y = solve_lower_triangular_system(L, b)
        return solve_upper_triangular_system(U, y)
    elif method == "LU_scipy_sparse":
        LU = splu(csc_matrix(A))
        return LU.solve(b)
    elif method == "QR":
        Q, R = QR_factorization(A.toarray())
        return solve_upper_triangular_system(R, np.dot(Q.T, b))
    elif method == "QR_compiled":
        Q, R = QR_factorization_compiled(A.toarray())
        return solve_upper_triangular_system(R, np.dot(Q.T, b))
    elif method == "QR_scipy":
        Q, R = qr(A.toarray())
        return solve_upper_triangular_system(R, np.dot(Q.T, b))
    elif method == "jacobi":
        return jacobi(A.toarray(), b, initial_guess, true_solution, tol)
    elif method == "jacobi_compiled":
        return jacobi_compiled(A.toarray(), b, initial_guess, true_solution, tol)
    elif method == "jacobi_sparse":
        return jacobi_sparse(A, b, initial_guess, true_solution, tol)
    elif method == "gauss_seidel":
        return gauss_seidel(A.toarray(), b, initial_guess, true_solution, tol)
    elif method == "gauss_seidel_compiled":
        return gauss_seidel_compiled(A.toarray(), b, initial_guess, true_solution, tol)
    elif method == "gauss_seidel_sparse":
        return gauss_seidel_sparse(A, b, initial_guess, true_solution, tol)
    elif method == "sor":
        return sor(A.toarray(), b, initial_guess, true_solution, tol, w)
    elif method == "sor_compiled":
        return sor_compiled(A.toarray(), b, initial_guess, true_solution, tol, w)
    elif method == "sor_sparse":
        return sor_sparse(A, b, initial_guess, true_solution, tol, w)
    elif method == "scipy":
        return solve(A.toarray(), b)


# Direct triangular system Solvers
def solve_upper_triangular_system(U, b):
    """ Solves the linear algebraic equation U*x = b, where b is a column
    vector, x is the unknown column vector and U is an upper triangular
    matrix.

    Parameters
    ----------
    U : array
        Square upper triangular matrix.
    b : array
        Column vector with the same number of lines as U.
    
    Returns
    -------
    x : array
        Column vector with the same number of lines as U which solves U*x=b.
    """
    # Retrieve matrix dimension
    n, m = U.shape

    # Initialize solution vector x:
    x = np.zeros(n)

    # Iterate line by line, beginning from the last one.
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - sum([U[i, j]*x[j] for j in range(i+1, n)]))/U[i, i]
    
    # Return the solution vector x
    return x

def solve_lower_triangular_system(L, b):
    """ Solves the linear algebraic equation L*y = b, where b is a column
    vector, y is the unknown column vector and L is a lower triangular
    matrix.

    Parameters
    ----------
    L : array
        Square lower triangular matrix.
    b : array
        Column vector with the same number of lines as L.
    
    Returns
    -------
    y : array
        Column vector with the same number of lines as L which solves L*y=b.
    """
    # Retrieve matrix dimension
    n, m = L.shape

    # Initialize solution vector y
    y = np.zeros(n)

    # Iterate line by line, beginning from the first one.
    for i in range(n):
        y[i] = (b[i] - sum([L[i, j]*y[j] for j in range(i)]))/L[i, i]
    
    # Return the solution vector y
    return y


# Factorization methods
def LU_factorization(A, return_L=False):
    """ Decomposes the given A matrix into its LU factors, that is A=LU,
    where L is a lower triangular matrix, with diagonal entries equal to 1 and
    U is an upper triangular matrix. By default, it returns U and L^(-1).

    Parameters
    ----------
    A : array
        Matrix to be decomposed into A=LU. A must be a square matrix with no
        diagonal entries equal to 0.
    return_L : bool, optional
        If True, returns U and L. If false, which it is by default, return U
        and L^(-1).

    Returns
    -------
    L : array
        Lower triangular matrix with the same shape as A, equal to L^(-1) if 
        return_L is False. If return_L is true, equal to L.
    U : array
        Upper triangular matrix with the same shape as A.

    """
    # Retrieve matrix dimension
    n, m = A.shape

    # Initialize L prime (L^(-1)) and U
    L_prime = np.eye(n, n, dtype=np.float64)
    U = A.copy()

    # Iterate column by column, except for the last column
    for j in range(m - 1):
        # Create temporary L_1
        L_temp = np.eye(n, n, dtype=np.float64)
        L_temp[(j+1):, j] = U[(j+1):, j]/(-U[j, j])

        # Multiply L_temp by current version of U to get new U
        U = L_temp.dot(U)

        # Multiply L_temp by current version of L_prime to get new L_prime
        L_prime = L_temp.dot(L_prime)
    
    # Return L and U
    if return_L:
        return inv(L_prime), U
    else:
        return L_prime, U

def LU_factorization_sparse(A, return_L=False):
    """ Decomposes the given A matrix into its LU factors, that is A=LU,
    where L is a lower triangular matrix, with diagonal entries equal to 1 and
    U is an upper triangular matrix. By default, it returns U and L^(-1). It
    makes use of sparse properties to optimize speed.

    Parameters
    ----------
    A : array
        Matrix to be decomposed into A=LU. A must be a square matrix with no
        diagonal entries equal to 0.
    return_L : bool, optional
        If True, returns U and L. If false, which it is by default, return U
        and L^(-1).

    Returns
    -------
    L : array
        Lower triangular matrix with the same shape as A, equal to L^(-1) if 
        return_L is False. If return_L is true, equal to L.
    U : array
        Upper triangular matrix with the same shape as A.

    """
    # Retrieve matrix dimension
    n, m = A.shape

    # Initialize L prime (L^(-1)) U
    L_prime = identity(n, np.float64, 'csr')
    U = csc_matrix(A)
    L_temp1 = identity(n, np.float64, 'lil')

    # Iterate column by column, except for the last column
    for j in range(m - 1):
        # Create temporary Ls
        L_temp1[(j+1):, j] = U[(j+1):, j]/(-U[j, j])
        L_temp2 = csr_matrix(L_temp1)

        # Roll back L_temp1
        L_temp1[(j+1):, j] = 0

        # Multiply L_temp by current version of U to get new U
        U = L_temp2.dot(U)

        # Multiply L_temp by current version of L_prime to get new L_prime
        L_prime = L_temp2.dot(L_prime)
    
    # Return L and U
    if return_L:
        return Inv(L_prime), U
    else:
        return L_prime, U

@jit(nopython=True)
def LU_factorization_compiled(A):
    """ Decomposes the given A matrix into its LU factors, that is A=LU,
    where L is a lower triangular matrix, with diagonal entries equal to 1 and
    U is an upper triangular matrix. The function is compiled the first time it
    is ran by numba.

    Parameters
    ----------
    A : array
        Matrix to be decomposed into A=LU. A must be a square matrix with no
        diagonal entries equal to 0 and in numpy format.
    return_L : bool, optional
        If True, returns U and L. If false, which it is by default, return U
        and L^(-1).

    Returns
    -------
    L : array
        Lower triangular matrix with the same shape as A.
    U : array
        Upper triangular matrix with the same shape as A.

    """
    # Retrieve matrix dimension
    n, m = A.shape

    # Initialize L and U
    L = np.eye(n, n)
    U = np.eye(n, n)

    # Create first column of L and first row of U
    L[:, 0] = A[:, 0]/A[0, 0]
    U[0, :] = A[0, :]

    # Iterate line by line, except for the first and last ones
    for i in range(1, n):
        #  Complete ith row of U and ith column of L
        for j in range(i, n):
            sum1 = 0
            for k in range(i):
                sum1 += L[i, k]*U[k, j]
            U[i, j] = A[i, j] - sum1
            sum2 = 0
            for k in range(i):
                sum2 += L[j, k]*U[k, i]
            L[j, i] = (A[j, i] - sum2)/U[i, i]
    
    # Return L and U
    return L, U

def QR_factorization(A):
    """Decomposes the given A matrix into its QR factors, that is A=QR,
    where Q is a orthogonal matrix and R is an upper triangular matrix.

    Parameters
    ----------
    A : array
        Matrix to be decomposed into A=QR. A must be a square matrix with no
        diagonal entries equal to 0.

    Returns
    -------
    Q : array
        Orthogonal matrix with the same shape as A.
    R : array
        Upper triangular matrix with the same shape as A.
    """
    # Retrieve matrix dimension
    n, m = A.shape

    # Initialize Q and R matrix
    Q = A.copy()
    R = np.eye(n, n)

    # Compute columns of Q matrix
    for j in range(n):
        # Subtract projections on previous columns
        temp1 = Q[:, j]
        temp2 = temp1 - sum([np.dot(temp1.T, Q[:, k])*Q[:, k] for k in range(j)])
        # Normalize result
        Q[:, j] = temp2/np.linalg.norm(temp2)

    # Compute lines of R matrix
    for i in range(n):
        R[i, i:] = [np.dot(A[:, j], Q[:, i]) for j in range(i, n)]
    
    # Return Q and R
    return Q, R

@jit(nopython=True)
def QR_factorization_compiled(A):
    """Decomposes the given A matrix into its QR factors, that is A=QR,
    where Q is a orthogonal matrix and R is an upper triangular matrix.

    Parameters
    ----------
    A : array
        Matrix to be decomposed into A=QR. A must be a square matrix with no
        diagonal entries equal to 0.

    Returns
    -------
    Q : array
        Orthogonal matrix with the same shape as A.
    R : array
        Upper triangular matrix with the same shape as A.
    """
    # Retrieve matrix dimension
    n, m = A.shape

    # Initialize Q and R matrix
    Q = A.copy()
    R = np.eye(n, n)

    # Compute columns of Q matrix
    for j in range(n):
        # Subtract projections on previous columns
        current_column = Q[:, j].copy()
        for k in range(j):
            reference_column = Q[:, k]
            # Project current column on reference column
            proj = 0
            for l in range(n):
                proj += current_column[l] * reference_column[l]
            # Subtract projection from current column
            current_column -= proj*reference_column
        # Calculate norm
        norm = 0.0
        for k in range(n):
            norm += current_column[k]**2
        norm = (norm)**0.5
        # Normalize result
        Q[:, j] = current_column/norm

    # Compute lines of R matrix
    for i in range(n):
        for j in range(i, n):
            # Scalar projection of jth column from A into ith column from Q
            proj = 0
            A_column = A[:, j]
            Q_column = Q[:, i]
            for k in range(n):
                proj += A_column[k] * Q_column[k]
            R[i, j] = proj

    # Return Q and R
    return Q, R


# Iterative Solvers
def jacobi(A, b, initial_guess, true_solution, tol):
    """ Solves the algebraic system A*x=b using a jacobi iterative solver.

    Parameters
    ----------
    A : array
        Matrix A corresponding to the system A*x = b.
    b : array
        Column vector b corresponding to the system A*x = b.
    initial_guess : array, optional
        Initial solution.
    true_solution : array, optional
        Used for residual calculation
    tol : float, optional
        Convergence tolerance.

    Returns
    -------
    x : array
        Solution vector corresponding to A*x = b.
    residue : array
        Residue array.
    """
    print("Initializing Jacobi Solver", end='\r')

    # Retrieve matrix dimension
    n, m = A.shape

    # Flatten b
    b = b.flatten()

    # Initialize u
    u = initial_guess.flatten()
    u_true = true_solution.flatten()
    u_new = u.copy()

    # Initialize residue
    res = np.linalg.norm(u - true_solution)
    res_array = [res]

    # Iterate
    while res > tol:
        print("Current Iteration and Residue: {:06d} - {:05.4E}".format(len(res_array), res), end='\r')
        
        for i in range(n):
            sum_factor = 0
            for j in range(i):
                sum_factor += A[i, j] * u[j]
            
            for j in range(i + 1, n):
                sum_factor += A[i, j] * u[j]

            u_new[i] = (1/A[i, i]) * (b[i] - sum_factor)
        
        # Update u
        u = u_new.copy()

        res = np.linalg.norm(u - true_solution)
        res_array += [res]
    
    print("\nJacobi solver converged after ", len(res_array), " iterations.")
    return u, res_array

@jit(nopython=True)
def jacobi_compiled(A, b, initial_guess, true_solution, tol):
    """ Solves the algebraic system A*x=b using a jacobi iterative solver.

    Parameters
    ----------
    A : array
        Matrix A corresponding to the system A*x = b.
    b : array
        Column vector b corresponding to the system A*x = b.
    initial_guess : array, optional
        Initial solution.
    true_solution : array, optional
        Used for residual calculation
    tol : float, optional
        Convergence tolerance.

    Returns
    -------
    x : array
        Solution vector corresponding to A*x = b.
    residue : array
        Residue array.
    """
    # Retrieve matrix dimension
    n, m = A.shape

    # Flatten b
    b = b.flatten()

    # Initialize u
    u = initial_guess.flatten()
    u_true = true_solution.flatten()
    u_new = u.copy()

    # Initialize residue
    res = np.linalg.norm(u - true_solution)
    res_array = [res]

    # Iterate
    while res > tol:
        for i in range(n):
            sum_factor = 0
            for j in range(i):
                sum_factor += A[i, j] * u[j]
            
            for j in range(i + 1, n):
                sum_factor += A[i, j] * u[j]

            u_new[i] = (1/A[i, i]) * (b[i] - sum_factor)
        
        # Update u
        u = u_new.copy()

        # Update residue
        res = np.linalg.norm(u - true_solution)
        res_array += [res]

    return u, res_array

def jacobi_sparse(A, b, initial_guess, true_solution, tol):
    """ Solves the algebraic system A*x=b using a jacobi iterative solver.

    Parameters
    ----------
    A : array
        Matrix A corresponding to the system A*x = b. Should be in sparse
        format.
    b : array
        Column vector b corresponding to the system A*x = b.
    initial_guess : array, optional
        Initial solution.
    true_solution : array, optional
        Used for residual calculation
    tol : float, optional
        Convergence tolerance.

    Returns
    -------
    x : array
        Solution vector corresponding to A*x = b.
    residue : array
        Residue array.
    """
    # print("Initializing Jacobi Sparse Solver", end='\r')
    # Retrieve matrix dimension
    n, m = A.shape
    
    # Define D by extracting main diagonal from A
    main_diagonal = A.diagonal()
    D = dia_matrix(([main_diagonal], [0]), shape=(n, n), dtype=np.float64)

    # Compute R
    R = A - D

    # Compute D^-1
    D_inverse = dia_matrix(([1/main_diagonal], [0]), shape=(n, n), dtype=np.float64)

    # Compute (D^-1)*b
    D_inverse_dot_b =  D_inverse.dot(b.flatten())

    # Compute -(D^-1)*R
    D_inverse_dot_R = D_inverse.dot(R)

    # Initialize u
    u = initial_guess.flatten()
    u_true = true_solution.flatten()

    # Initialize residue
    res = np.linalg.norm(u - true_solution)
    res_array = [res]

    # Iterate
    while res > tol:
        # print("Current Iteration and Residue: {:06d} - {:05.4E}".format(len(res_array), res), end='\r')

        # Update u
        u = D_inverse_dot_b - D_inverse_dot_R.dot(u)

        # Update residue
        res = np.linalg.norm(u - true_solution)
        res_array += [res]
    
    # print("\nJacobi solver converged after ", len(res_array), " iterations.")
    return u, res_array

def gauss_seidel(A, b, initial_guess, true_solution, tol):
    """ Solves the algebraic system A*x=b using a Gauss-Seidel iterative solver.

    Parameters
    ----------
    A : array
        Matrix A corresponding to the system A*x = b.
    b : array
        Column vector b corresponding to the system A*x = b.
    initial_guess : array, optional
        Initial solution.
    true_solution : array, optional
        Used for residual calculation
    tol : float, optional
        Convergence tolerance.

    Returns
    -------
    x : array
        Solution vector corresponding to A*x = b.
    residue : array
        Residue array.
    """
    print("Initializing Gauss-Seidel Solver", end='\r')

    # Retrieve matrix dimension
    n, m = A.shape

    # Flatten b
    b = b.flatten()

    # Initialize u
    u = initial_guess.flatten()
    u_true = true_solution.flatten()

    # Initialize residue
    res = np.linalg.norm(u - true_solution)
    res_array = [res]

    # Iterate
    while res > tol:
        print("Current Iteration and Residue: {:06d} - {:05.4E}".format(len(res_array), res), end='\r')
        
        for i in range(n):
            sum_factor = 0
            for j in range(i):
                sum_factor += A[i, j] * u[j]
            
            for j in range(i + 1, n):
                sum_factor += A[i, j] * u[j]

            u[i] = (1/A[i, i]) * (b[i] - sum_factor)
        
        res = np.linalg.norm(u - true_solution)
        res_array += [res]
    
    print("\nGauss-Seidel solver converged after ", len(res_array), " iterations.")
    return u, res_array

@jit(nopython=True)
def gauss_seidel_compiled(A, b, initial_guess, true_solution, tol):
    """ Solves the algebraic system A*x=b using a Gauss-Seidel iterative solver.
    This function takes advantage of just in time compilation to speed up
    evaluations.

    Parameters
    ----------
    A : array
        Matrix A corresponding to the system A*x = b.
    b : array
        Column vector b corresponding to the system A*x = b.
    initial_guess : array, optional
        Initial solution.
    true_solution : array, optional
        Used for residual calculation
    tol : float, optional
        Convergence tolerance.

    Returns
    -------
    x : array
        Solution vector corresponding to A*x = b.
    residue : array
        Residue array.
    """
    # Retrieve matrix dimension
    n, m = A.shape

    # Flatten b
    b = b.flatten()

    # Initialize u
    u = initial_guess.flatten()
    u_true = true_solution.flatten()

    # Initialize residue
    res = np.linalg.norm(u - true_solution)
    res_array = [res]

    # Iterate
    while res > tol:
        for i in range(n):
            sum_factor = 0
            for j in range(i):
                sum_factor += A[i, j] * u[j]
            
            for j in range(i + 1, n):
                sum_factor += A[i, j] * u[j]

            u[i] = (1/A[i, i]) * (b[i] - sum_factor)

        # Update residue
        res = np.linalg.norm(u - true_solution)
        res_array += [res]

    return u, res_array

def gauss_seidel_sparse(A, b, initial_guess, true_solution, tol):
    """ Solves the algebraic system A*x=b using a Gauss-Seidel iterative solver.
    This function makes use of the sparsity of the A matrix to speed up
    function evaluation.

    Parameters
    ----------
    A : array
        Matrix A corresponding to the system A*x = b. Should be in sparse
        format.
    b : array
        Column vector b corresponding to the system A*x = b.
    initial_guess : array, optional
        Initial solution.
    true_solution : array, optional
        Used for residual calculation
    tol : float, optional
        Convergence tolerance.

    Returns
    -------
    x : array
        Solution vector corresponding to A*x = b.
    residue : array
        Residue array.
    """
    # print("Initializing Gauss-Seidel Sparse Solver", end='\r')
    # Retrieve matrix dimension
    n, m = A.shape
    
    # Compute L by extracting lower triangular matrix from A
    L = tril(A)
    
    # Compute U by extracting upper triangular matrix from A
    U = triu(A, k=1)

    # Compute L^-1
    L_inverse = sparse_inv(csc_matrix(L))

    # Compute (L^-1)*b
    L_inverse_dot_b =  L_inverse.dot(b.flatten())

    # Compute -(L^-1)*U
    L_inverse_dot_U = L_inverse.dot(U)

    # Initialize u
    u = initial_guess.flatten()
    u_true = true_solution.flatten()

    # Initialize residue
    res = np.linalg.norm(u - true_solution)
    res_array = [res]

    # Iterate
    while res > tol:
        # print("Current Iteration and Residue: {:06d} - {:05.4E}".format(len(res_array), res), end='\r')

        # Update u
        u = L_inverse_dot_b - L_inverse_dot_U.dot(u)

        # Update residue
        res = np.linalg.norm(u - true_solution)
        res_array += [res]
    
    # print("\nGauss-Seidel solver converged after ", len(res_array), " iterations.")
    return u, res_array

def sor(A, b, initial_guess, true_solution, tol, w):
    """ Solves the algebraic system A*x=b using a Gauss-Seidel Successive
    Over Relaxation iterative solver.

    Parameters
    ----------
    A : array
        Matrix A corresponding to the system A*x = b.
    b : array
        Column vector b corresponding to the system A*x = b.
    initial_guess : array, optional
        Initial solution.
    true_solution : array, optional
        Used for residual calculation
    tol : float, optional
        Convergence tolerance.
    w : float
        Relaxation parameter.

    Returns
    -------
    x : array
        Solution vector corresponding to A*x = b.
    residue : array
        Residue array.
    """
    print("Initializing Gauss-Seidel SOR Solver", end='\r')

    # Retrieve matrix dimension
    n, m = A.shape

    # Flatten b
    b = b.flatten()

    # Initialize u
    u = initial_guess.flatten()
    u_true = true_solution.flatten()

    # Initialize residue
    res = np.linalg.norm(u - true_solution)
    res_array = [res]

    # Iterate
    while res > tol:
        print("Current Iteration and Residue: {:06d} - {:05.4E}".format(len(res_array), res), end='\r')
        
        for i in range(n):
            sum_factor = 0
            for j in range(i):
                sum_factor += A[i, j] * u[j]
            
            for j in range(i + 1, n):
                sum_factor += A[i, j] * u[j]

            u[i] = (1-w)*u[i] + w*(1/A[i, i]) * (b[i] - sum_factor)
        
        res = np.linalg.norm(u - true_solution)
        res_array += [res]
    
    print("\nGauss-Seidel Successive Over Relaxation solver converged after ", len(res_array), " iterations.")
    return u, res_array

@jit(nopython=True)
def sor_compiled(A, b, initial_guess, true_solution, tol, w):
    """ Solves the algebraic system A*x=b using a Gauss-Seidel Successive
    Over Relaxation iterative solver. This function takes advantage of just
    in time compilation to speed up evaluations.

    Parameters
    ----------
    A : array
        Matrix A corresponding to the system A*x = b.
    b : array
        Column vector b corresponding to the system A*x = b.
    initial_guess : array, optional
        Initial solution.
    true_solution : array, optional
        Used for residual calculation
    tol : float, optional
        Convergence tolerance.
    w : float
        Relaxation parameter.

    Returns
    -------
    x : array
        Solution vector corresponding to A*x = b.
    residue : array
        Residue array.
    """
    # Retrieve matrix dimension
    n, m = A.shape

    # Flatten b
    b = b.flatten()

    # Initialize u
    u = initial_guess.flatten()
    u_true = true_solution.flatten()

    # Initialize residue
    res = np.linalg.norm(u - true_solution)
    res_array = [res]

    # Iterate
    while res > tol:
        for i in range(n):
            sum_factor = 0
            for j in range(i):
                sum_factor += A[i, j] * u[j]
            
            for j in range(i + 1, n):
                sum_factor += A[i, j] * u[j]

            u[i] = (1-w)*u[i] + w*(1/A[i, i]) * (b[i] - sum_factor)
        
        res = np.linalg.norm(u - true_solution)
        res_array += [res]

    return u, res_array

def sor_sparse(A, b, initial_guess, true_solution, tol, w):
    """ Solves the algebraic system A*x=b using a Gauss-Seidel Successive
    Over Relaxation iterative solver. This function makes use of the sparsity
    of the A matrix to speed up function evaluation.

    Parameters
    ----------
    A : array
        Matrix A corresponding to the system A*x = b. Should be in sparse
        format.
    b : array
        Column vector b corresponding to the system A*x = b.
    initial_guess : array, optional
        Initial solution.
    true_solution : array, optional
        Used for residual calculation
    tol : float, optional
        Convergence tolerance. 
    w : float
        Relaxation parameter.

    Returns
    -------
    x : array
        Solution vector corresponding to A*x = b.
    residue : array
        Residue array.
    """
    # print("Initializing Gauss-Seidel Successive Over Relaxation Sparse Solver", end='\r')
    # Retrieve matrix dimension
    n, m = A.shape

    # Compute D by extracting main diagonal from A
    main_diagonal = A.diagonal()
    D = dia_matrix(([main_diagonal], [0]), shape=(n, n), dtype=np.float64)

    # Compute L by extracting lower triangular matrix from A
    L = tril(A, k=-1)
    
    # Compute U by extracting upper triangular matrix from A
    U = triu(A, k=1)

    # Compute (D + w*L)^-1
    D_plus_w_L_inverse = sparse_inv(csc_matrix(D + w*L))

    # Compute w*b
    w_b =  w*b.flatten()

    # Compute w*U + (w-1)*D
    w_U_plus_w_minus_1_D = w*U + (w-1)*D

    # Compute ((D + w*L)^-1)*(w*U + (w-1)*D)
    D_plus_w_L_inverse_dot_w_U_plus_w_minus_1_D = D_plus_w_L_inverse.dot(w_U_plus_w_minus_1_D)

    # Compute ((D + w*L)^-1)*w*b0
    D_plus_w_L_inverse_dot_w_b = D_plus_w_L_inverse.dot(w_b)

    # Initialize u
    u = initial_guess.flatten()
    u_true = true_solution.flatten()

    # Initialize residue
    res = np.linalg.norm(u - true_solution)
    res_array = [res]

    # Iterate
    while res > tol:
        # print("Current Iteration and Residue: {:06d} - {:05.4E}".format(len(res_array), res), end='\r')

        # Update u
        u = D_plus_w_L_inverse_dot_w_b - D_plus_w_L_inverse_dot_w_U_plus_w_minus_1_D.dot(u)

        # Update residue
        res = np.linalg.norm(u - true_solution)
        res_array += [res]
    
    # print("\nGauss-Seidel Successive Over Relaxation solver converged after ", len(res_array), " iterations.")
    return u, res_array
