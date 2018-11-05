# Assignment 1 for ME4233
# Author: Giovani Hidalgo Ceotto
# Prof: Mengqi Zhang

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from itertools import product
from scipy.sparse import dia_matrix, csr_matrix, csc_matrix, lil_matrix, identity, tril, triu

# Main functions used
def assemble_algebraic_system(Lx, Ly, grid_points_x, grid_points_y, g, boundary_conditions):
    """ Assemble A matrix and b vector, corresponding to the the A*u = b
    system which results from 5 point finite difference discretization
    of the partial differential equation: ((d2/dx2) + (d2/dy2))*u = g,
    on a rectangular domain.

    Parameters
    ----------
    Lx : int, float
        X side length of the domain.
    Ly : int, float
        Y side length of the domain.
    grid_points_x : int
        Number of grid points to discretize the domain in the x axis.
    grid_points_y : int
        Number of grid points to discretize the domain in the y axis.
    g : function
        Function of x and y which gives the source term of the differential
        equation.
    boundary_conditions : function
        Function of x and y, defined in the boundary of the domain, which
        returns the value of u in the boundary.

    Returns
    -------
    A : array
        Diagonal sparse A matrix corresponding to the the A*u = b system which results from 5
        point finite difference discretization of the partial differential
        equation: ((d2/dx2) + (d2/dy2))*u = g, on a rectangular domain, 
        assuming the u vector is arranged as u=[u11, u12, ...].
    b :  array
        b vector corresponding to the the A*u = b system which results from 5
        point finite difference discretization of the partial differential
        equation: ((d2/dx2) + (d2/dy2))*u = g, on a rectangular domain, 
        assuming the u vector is arranged as u=[u11, u12, ...].
    """
    # Calculate number of divisions between grid points
    Nx = grid_points_x - 1
    Ny = grid_points_y - 1

    # Calculate dx and dy
    dx = Lx/Nx
    dy = Ly/Ny

    # Calculate A matrix size
    A_size = (Nx - 1) * (Ny - 1)

    # Calculate recurrent multipliers
    a = (1/(dy**2))
    c = (1/(dx**2))
    b = (-2*a - 2*c)

    # Generate primary diagonal
    primary_dia = b*np.ones(A_size)

    # Generate secondary diagonal
    secondary_dia = (Ny - 2)*[a] + [0]
    secondary_dia_upper = [0] + (Nx - 2)*secondary_dia + (Ny - 2)*[a] # Left padding with 0 
    secondary_dia_upper = np.array(secondary_dia_upper)
    secondary_dia_lower = (Nx - 1)*secondary_dia
    secondary_dia_lower = np.array(secondary_dia_lower)

    # Generate tertiary diagonal
    tertiary_dia = c*np.ones(A_size)

    # Create A matrix
    data = np.array([tertiary_dia, secondary_dia_lower, primary_dia, secondary_dia_upper, tertiary_dia])
    A = dia_matrix((data, [-(Ny-1), -1, 0, 1, (Ny-1)]), shape=(A_size, A_size), dtype=np.float64)

    # Initialize b vector
    i_list = range(1, Nx)
    j_list = range(1, Ny)
    b = np.array([g(i*dx, j*dy) for i, j in product(i_list, j_list)])

    # Subtract boundary condition from B vector

    # Top and bottom
    for i in range(1, Nx):
        # Bottom boundary condition
        j = 1
        line_index = (i - 1)*(Ny - 1) + (j - 1)
        b[line_index] -= a*boundary_conditions(i*dx, 0)
        # Top boundary condition
        j = Ny - 1
        line_index = (i - 1)*(Ny - 1) + (j - 1)
        b[line_index] -= a*boundary_conditions(i*dx, Ly)
    
    # Left and right
    for j in range(1, Ny):
        # Left boundary condition
        i = 1
        line_index = (i - 1)*(Ny - 1) + (j - 1)
        b[line_index] -= c*boundary_conditions(0, j*dy)
        # Right boundary condition
        i = Nx - 1
        line_index = (i - 1)*(Ny - 1) + (j - 1)
        b[line_index] -= c*boundary_conditions(Lx, j*dy)

    # Return A matrix and b vector
    return A, b
    
def post_process_solution(Lx, Ly, grid_points_x, grid_points_y, solution_vector, boundary_conditions):
    """ Post process the solution vector, by adding boundary conditions and
    plotting the solution as a surface in a 3D plot and as a contours in a 
    2D plot.

    Parameters
    ----------
    Lx : int, float
        X side length of the domain.
    Ly : int, float
        Y side length of the domain.
    grid_points_x : int
        Number of grid points used to discretize the domain in the x axis.
    grid_points_y : int
        Number of grid points used to discretize the domain in the y axis.
    solution_vector : array
        Vector which results from 'solve_algebraic_system', representing
        the solution of a partial differential equation on the discretized
        rectangular domain. Should be arranged as u=[u11, u12, ...]
    boundary_conditions : function
        Function of x and y, defined in the boundary of the domain, which
        returns the value of u in the boundary.

    Returns
    -------
    solution_matrix : array
        Represents the solution vector and the boundary conditions on a 
        matrix in which each entry corresponds to the solution in the mesh
        used.
    """
    # Calculate number of divisions between grid points
    Nx = grid_points_x - 1
    Ny = grid_points_y - 1

    # Calculate dx and dy
    dx = Lx/Nx
    dy = Ly/Ny

    # Convert solution vector to matrix mesh form
    solution_matrix = solution_vector.reshape(grid_points_x - 2,
                                              grid_points_y -2)

    # Pad solution matrix to make space for boundary condition
    solution_matrix = np.pad(solution_matrix, 1, 'constant', constant_values=0)

    # Populate solution matrix with boundary condition
    # Top and bottom
    for i in range(grid_points_x):
        # Bottom boundary condition
        solution_matrix[i, 0] = boundary_conditions(i*dx, 0)
        # Top boundary condition
        solution_matrix[i, -1] = boundary_conditions(i*dx, Ly)
    
    # Left and right
    for j in range(1, grid_points_y - 1):
        # Left boundary condition
        solution_matrix[0, j] = boundary_conditions(0, j*dy)
        # Right boundary condition
        solution_matrix[-1, j] = boundary_conditions(-1, j*dy)

    # Generate plot mesh
    x_mesh = np.linspace(0, Lx, grid_points_x)
    y_mesh = np.linspace(0, Ly, grid_points_y)
    X_mesh, Y_mesh = np.meshgrid(x_mesh, y_mesh)
    X_mesh = X_mesh.T
    Y_mesh = Y_mesh.T

    # Plot solution contour
    fig = plt.figure(figsize=(9, 4))
    ax2D = fig.add_subplot(1, 2, 1)
    contour = ax2D.contourf(X_mesh, Y_mesh, solution_matrix)
    fig.colorbar(contour, aspect=10)
    ax2D.set_xlabel('$x$')
    ax2D.set_ylabel('$y$')
    ax2D.set_title('$u(x, y)$')

    # Plot soultion surface
    ax3D = fig.add_subplot(1, 2, 2, projection='3d')
    ax3D.plot_surface(X_mesh, Y_mesh, solution_matrix, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax3D.set_xlabel('$x$')
    ax3D.set_ylabel('$y$')
    ax3D.set_zlabel('$u(x, y)$')
    plt.show()

    return solution_matrix

# Other unused functions (less efficient)
def assemble_algebraic_system_slow(Lx, Ly, grid_points_x, grid_points_y, g, boundary_conditions):
    """ Assemble A matrix and b vector, corresponding to the the A*u = b
    system which results from 5 point finite difference discretization
    of the partial differential equation: ((d2/dx2) + (d2/dy2))*u = g,
    on a rectangular domain.

    Parameters
    ----------
    Lx : int, float
        X side length of the domain.
    Ly : int, float
        Y side length of the domain.
    grid_points_x : int
        Number of grid points to discretize the domain in the x axis.
    grid_points_y : int
        Number of grid points to discretize the domain in the y axis.
    g : function
        Function of x and y which gives the source term of the differential
        equation.
    boundary_conditions : function
        Function of x and y, defined in the boundary of the domain, which
        returns the value of u in the boundary.

    Returns
    -------
    A : array
        A matrix corresponding to the the A*u = b system which results from 5
        point finite difference discretization of the partial differential
        equation: ((d2/dx2) + (d2/dy2))*u = g, on a rectangular domain, 
        assuming the u vector is arranged as u=[u11, u12, ...].
    b :  array
        b vector corresponding to the the A*u = b system which results from 5
        point finite difference discretization of the partial differential
        equation: ((d2/dx2) + (d2/dy2))*u = g, on a rectangular domain, 
        assuming the u vector is arranged as u=[u11, u12, ...].
    """
    # Compute Nx and Ny - number of divisions
    Nx = grid_points_x - 1
    Ny = grid_points_y - 1

    # Compute matrix size
    num_of_nodes = (Nx-1)*(Ny-1)

    # Calculate auxiliary values
    dx = Lx/Nx
    dy = Ly/Ny
    dx2 = dx**2
    dy2 = dy**2

    # Calculate recurrent multipliers
    a = (1/(dy**2))
    c = (1/(dx**2))
    b = (-2*a - 2*c)

    # Initialize A matrix
    A = lil_matrix((num_of_nodes, num_of_nodes), dtype=np.float64)

    # Iterate over interior points (not edges or corners)
    i_list = range(2, Nx-1)
    j_list = range(2, Ny-1)
    for i, j in product(i_list, j_list):
        # Calculate index of line to populate
        line_index = (i - 1)*(Ny - 1) + (j - 1)
        # Populate matrix entry for u_(i,j)
        A[line_index, line_index] = b
        # Populate matrix entry for u_(i,j+1)
        A[line_index, line_index + 1] = a
        # Populate matrix entry for u_(i,j-1)
        A[line_index, line_index - 1] = a
        # Populate matrix entry for u_(i+1,j)
        A[line_index, line_index + (Ny - 1)] = c
        # Populate matrix entry for u_(i-1,j)
        A[line_index, line_index - (Ny - 1)] = c

    # Check structure of A
    A1 = A - 0*A

    # Iterate over interior edge points (not corners)
    # Bottom Edge
    j = 1
    for i in i_list:
        # Calculate index of line to populate
        line_index = (i - 1)*(Ny - 1)
        # Populate matrix entry for u_(i,j)
        A[line_index, line_index] = b
        # Populate matrix entry for u_(i,j+1)
        A[line_index, line_index + 1] = a
        # Populate matrix entry for u_(i+1,j)
        A[line_index, line_index + (Ny - 1)] = c
        # Populate matrix entry for u_(i-1,j)
        A[line_index, line_index - (Ny - 1)] = c
    # Bottom Edge
    j = Ny - 1
    for i in i_list:
        # Calculate index of line to populate
        line_index = (i - 1)*(Ny - 1) + (Ny - 2)
        # Populate matrix entry for u_(i,j)
        A[line_index, line_index] = b
        # Populate matrix entry for u_(i,j-1)
        A[line_index, line_index - 1] = a
        # Populate matrix entry for u_(i+1,j)
        A[line_index, line_index + (Ny - 1)] = c
        # Populate matrix entry for u_(i-1,j)
        A[line_index, line_index - (Ny - 1)] = c
    # Left Edge
    i = 1
    for j in j_list:
        # Calculate index of line to populate
        line_index = (j - 1)
        # Populate matrix entry for u_(i,j)
        A[line_index, line_index] = b
        # Populate matrix entry for u_(i,j+1)
        A[line_index, line_index + 1] = a
        # Populate matrix entry for u_(i,j-1)
        A[line_index, line_index - 1] = a
        # Populate matrix entry for u_(i+1,j)
        A[line_index, line_index + (Ny - 1)] = c
    # Right Edge
    i = Nx - 1
    for j in j_list:
        # Calculate index of line to populate
        line_index = (Nx - 2)*(Ny - 1) + (j - 1)
        # Populate matrix entry for u_(i,j)
        A[line_index, line_index] = b
        # Populate matrix entry for u_(i,j+1)
        A[line_index, line_index + 1] = a
        # Populate matrix entry for u_(i,j-1)
        A[line_index, line_index - 1] = a
        # Populate matrix entry for u_(i-1,j)
        A[line_index, line_index - (Ny - 1)] = c

    # Check structure of A
    A2 = A[:, :] - A1

    # Iterate over corners
    # Bottom-Left
    i, j = 1, 1
    # Calculate index of line to populate
    line_index = (i - 1)*(Ny - 1) + (j - 1)
    # Populate matrix entry for u_(i,j)
    A[line_index, line_index] = b
    # Populate matrix entry for u_(i,j+1)
    A[line_index, line_index + 1] = a
    # Populate matrix entry for u_(i+1,j)
    A[line_index, line_index + (Ny - 1)] = c
    # Top-Left
    i, j = 1, Ny - 1
    # Calculate index of line to populate
    line_index = (i - 1)*(Ny - 1) + (j - 1)
    # Populate matrix entry for u_(i,j)
    A[line_index, line_index] = b
    # Populate matrix entry for u_(i,j-1)
    A[line_index, line_index - 1] = a
    # Populate matrix entry for u_(i+1,j)
    A[line_index, line_index + (Ny - 1)] = c
    # Top-Right
    i, j = Nx - 1, Ny - 1
    # Calculate index of line to populate
    line_index = (i - 1)*(Ny - 1) + (j - 1)
    # Populate matrix entry for u_(i,j)
    A[line_index, line_index] = b
    # Populate matrix entry for u_(i,j-1)
    A[line_index, line_index - 1] = a
    # Populate matrix entry for u_(i-1,j)
    A[line_index, line_index - (Ny - 1)] = c
    # Bottom-Right
    i, j = Nx - 1, 1
    # Calculate index of line to populate
    line_index = (i - 1)*(Ny - 1) + (j - 1)
    # Populate matrix entry for u_(i,j)
    A[line_index, line_index] = b
    # Populate matrix entry for u_(i,j+1)
    A[line_index, line_index + 1] = a
    # Populate matrix entry for u_(i-1,j)
    A[line_index, line_index - (Ny - 1)] = c

    # Convert A matrix to csr sparse
    A = csr_matrix(A)

    # Initialize b vector
    i_list = range(1, Nx)
    j_list = range(1, Ny)
    b = np.array([g(i*dx, j*dy) for i, j in product(i_list, j_list)])

    # Subtract boundary condition from B vector
    
    # Top and bottom
    for i in range(1, Nx):
        # Bottom boundary condition
        j = 1
        line_index = (i - 1)*(Ny - 1) + (j - 1)
        b[line_index] -= a*boundary_conditions(i*dx, 0)
        # Top boundary condition
        j = Ny - 1
        line_index = (i - 1)*(Ny - 1) + (j - 1)
        b[line_index] -= a*boundary_conditions(i*dx, Ly)
    
    # Left and right
    for j in range(1, Ny):
        # Left boundary condition
        i = 1
        line_index = (i - 1)*(Ny - 1) + (j - 1)
        b[line_index] -= c*boundary_conditions(0, j*dy)
        # Right boundary condition
        i = Nx - 1
        line_index = (i - 1)*(Ny - 1) + (j - 1)
        b[line_index] -= c*boundary_conditions(Lx, j*dy)

        # Analyze A building blocks
        # plt.figure()
        # plt.spy(A1, markersize=5, color='green')
        # plt.spy(A2, markersize=5, color='blue')
        # plt.spy(A-A2-A1, markersize=5, color='red')
        # plt.show()

    # Return A matrix and b vector
    return A, b