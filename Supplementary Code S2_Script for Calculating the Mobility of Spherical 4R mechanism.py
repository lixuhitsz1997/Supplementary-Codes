import sympy as sp

# Define the symbolic variables
d2, d3, d4, e1, e3, e4, f3, f4 = sp.symbols('d2 d3 d4 e1 e3 e4 f3 f4')

# Define the position vectors for each joint
r1 = sp.Matrix([0, e1, 0])
r2 = sp.Matrix([d2, 0, 0])
r3 = sp.Matrix([d3, e3, f3])
r4 = sp.Matrix([d4, e4, f4])

# Define the direction vectors for each joint (from joint position to the origin)
L1 = -r1
L2 = -r2
L3 = -r3
L4 = -r4

# Normalize the direction vectors
L1 = L1 / L1.norm()
L2 = L2 / L2.norm()
L3 = L3 / L3.norm()
L4 = L4 / L4.norm()

# Define the motion screws for each joint
def motion_screw(L, r):
    M = r.cross(L)
    return sp.Matrix.vstack(L, M)

S1 = motion_screw(L1, r1)
S2 = motion_screw(L2, r2)
S3 = motion_screw(L3, r3)
S4 = motion_screw(L4, r4)

# Concatenate the motion screws for each branch
branch1_screws = sp.Matrix.hstack(S1, S4)
branch2_screws = sp.Matrix.hstack(S2, S3)

# Compute the reciprocal screw system for each branch
def reciprocal_screw_system(screws):
    screws_T = screws.T
    null_space = screws_T.nullspace()
    return sp.Matrix.hstack(*null_space)

branch1_constraint_screws = reciprocal_screw_system(branch1_screws)
branch2_constraint_screws = reciprocal_screw_system(branch2_screws)

# Concatenate the constraint screws from all branches
platform_constraint_screws = sp.Matrix.hstack(branch1_constraint_screws, branch2_constraint_screws)

# Compute the motion screw system of the moving platform
platform_motion_screws = reciprocal_screw_system(platform_constraint_screws)

# Print the motion screw system of the moving platform
sp.pprint(platform_motion_screws)
