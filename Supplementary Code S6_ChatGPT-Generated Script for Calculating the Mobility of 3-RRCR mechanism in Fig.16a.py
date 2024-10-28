import sympy as sp

# Define symbolic variables for the position vectors and directions
g1, h1, g2, h2, g3, h3, k3, g4, h4 = sp.symbols('g1 h1 g2 h2 g3 h3 k3 g4 h4')
g5, h5, g6, h6, g7, h7, k7, g8, h8 = sp.symbols('g5 h5 g6 h6 g7 h7 k7 g8 h8')
g9, h9, g10, h10, g11, h11, k11, g12, h12 = sp.symbols('g9 h9 g10 h10 g11 h11 k11 g12 h12')

# Define the direction vectors for the rotational joints (parallel to z-axis)
Lz = sp.Matrix([0, 0, 1])

# Define the motion screws for the rotational joints
def rotational_screw(position, direction):
    r = sp.Matrix(position)
    L = sp.Matrix(direction)
    M = r.cross(L)
    return sp.Matrix.vstack(L, M)

# Define the motion screws for the prismatic joints
def prismatic_screw(direction):
    L = sp.Matrix(direction)
    return sp.Matrix.vstack(sp.zeros(3, 1), L)

# Define the motion screws for the cylindrical joints
def cylindrical_screw(position, direction):
    r = sp.Matrix(position)
    L = sp.Matrix(direction)
    M = r.cross(L)
    return sp.Matrix.hstack(sp.Matrix.vstack(L, M), sp.Matrix.vstack(sp.zeros(3, 1), L))

# Branch 1
S1_R1 = rotational_screw([g1, h1, 0], Lz)
S1_R2 = rotational_screw([g2, h2, 0], Lz)
S1_R3 = rotational_screw([g3, h3, k3], [-g3, -h3, -k3])
S1_C4 = cylindrical_screw([g4, h4, 0], [-g4, -h4, 0])

# Branch 2
S2_R5 = rotational_screw([g5, h5, 0], Lz)
S2_R6 = rotational_screw([g6, h6, 0], Lz)
S2_R7 = rotational_screw([g7, h7, k7], [-g7, -h7, -k7])
S2_C8 = cylindrical_screw([g8, h8, 0], [-g8, -h8, 0])

# Branch 3
S3_R9 = rotational_screw([g9, h9, 0], Lz)
S3_R10 = rotational_screw([g10, h10, 0], Lz)
S3_R11 = rotational_screw([g11, h11, k11], [-g11, -h11, -k11])
S3_C12 = cylindrical_screw([g12, h12, 0], [-g12, -h12, 0])

# Combine the motion screws for each branch
S1 = sp.Matrix.hstack(S1_R1, S1_R2, S1_R3, S1_C4)
S2 = sp.Matrix.hstack(S2_R5, S2_R6, S2_R7, S2_C8)
S3 = sp.Matrix.hstack(S3_R9, S3_R10, S3_R11, S3_C12)

# Compute the constraint-screw system for each branch (reciprocal screws)
C1 = S1.T.nullspace()
C2 = S2.T.nullspace()
C3 = S3.T.nullspace()

# Combine the constraint-screw systems from all branches
C = sp.Matrix.hstack(*C1, *C2, *C3)

# Compute the motion-screw system of the moving platform (reciprocal of the constraint-screw system)
M = C.T.nullspace()

# Print the motion-screw system of the moving platform
for i, screw in enumerate(M):
    print(f"Motion Screw {i+1}:")
    sp.pprint(screw)
    print()