import numpy as np
import dolfinx
from dolfinx import geometry
import basix


def interpolation_matrix_nonmatching_meshes(V_1,V_0): # Function spaces from nonmatching meshes
    msh_0 = V_0.mesh
    msh_1 = V_1.mesh
    x_0   = V_0.tabulate_dof_coordinates()
    x_1   = V_1.tabulate_dof_coordinates()

    bb_tree         = geometry.BoundingBoxTree(msh_0, msh_0.topology.dim)
    cell_candidates = geometry.compute_collisions(bb_tree, x_1)
    cells           = []
    points_on_proc  = []
    index_points    = []
    colliding_cells = geometry.compute_colliding_cells(msh_0, cell_candidates, x_1)

    for i, point in enumerate(x_1):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
            index_points.append(i)
            
    index_points_   = np.array(index_points)
    points_on_proc_ = np.array(points_on_proc, dtype=np.float64)
    cells_          = np.array(cells)

    ct      = dolfinx.cpp.mesh.to_string(msh_0.topology.cell_type)
    element = basix.create_element(basix.finite_element.string_to_family(
        "Lagrange", ct), basix.cell.string_to_type(ct), V_0.ufl_element().degree(), basix.LagrangeVariant.equispaced)

    x_ref = np.zeros((len(cells_), 3))

    for i in range(0, len(cells_)):
        geom_dofs  = msh_0.geometry.dofmap.links(cells_[i])
        x_ref[i,:] = msh_0.geometry.cmap.pull_back([points_on_proc_[i,:]], msh_0.geometry.x[geom_dofs])

    basis_matrix = element.tabulate(0, x_ref)[0,:,:,0]

    cell_dofs         = np.zeros((len(x_1), len(basis_matrix[0,:])))
    basis_matrix_full = np.zeros((len(x_1), len(basis_matrix[0,:])))


    for nn in range(0,len(cells_)):
        cell_dofs[index_points_[nn],:] = V_0.dofmap.cell_dofs(cells_[nn])
        basis_matrix_full[index_points_[nn],:] = basis_matrix[nn,:]

    cell_dofs_ = cell_dofs.astype(int) ###### REDUCE HERE

    I = np.zeros((len(x_1), len(x_0)), dtype=complex)

    for i in range(0,len(x_1)):
        for j in range(0,len(basis_matrix[0,:])):
            I[i,cell_dofs_[i,j]] = basis_matrix_full[i,j]

    return I 

