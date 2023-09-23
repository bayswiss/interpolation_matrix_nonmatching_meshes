# interpolation_matrix_nonmatching_meshes
Creates a numpy interpolation matrix between two FunctionSpaces coming from non-matching meshes.

Tested on dolfinx v0.6.0
Works only for lagrange elements.

Suppose that we have the following situation:
```
V_0 = FunctionSpace(mesh_0,("CG",deg_0))
V_1 = FunctionSpace(mesh_1,("CG",deg_1))

p_0 = Function(V_0)
...
p_0 = SomeComputations()
```

we define a new function on $V_1$:
```
p_1 = Function(V_1)
```
If we wanted to interpolate the function, we could use the command:
```
p_1.interpolate(p_0)
```
But now we have an alternative. We can indeed compute the interpolation matrix and compute the `p_1` array from a matrix multiplication between `I_matrix` and the `p_0` array:
```
I_matrix = interpolation_matrix_nonmatching_meshes(V_1,V_0)
p_1.x.array[:] = np.matmul(I_matrix,p_0.x.array)
```
