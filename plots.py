import numpy as np
import matplotlib.pyplot as plt
from functions import registry

def surface_grid(function, lb: np.ndarray, ub: np.ndarray, grid_points: int=120):
    """Generate a grid for surface/contour plots of a 2D function."""
    # create evenly spaced coordinates in each dimension between bounds
    x = np.linspace(lb[0], ub[0], grid_points)
    y = np.linspace(lb[1], ub[1], grid_points)
    # create 2D grid from 1D coordinate arrays
    X, Y = np.meshgrid(x, y)
    # stack grid points into shape (N, 2) for vectorized function evaluation
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    # evaluate function at all grid points and reshape back to grid shape
    Z = function(pts).reshape(X.shape)
    return X, Y, Z

def plot_surface(fn_name: str, grid=200, elev=45, azim=45, cmap='viridis', surf_alpha=0.85):
    """3D surface plot for a 2D benchmark function in registry."""
    # lookup benchmark function object from registry
    bf = registry[fn_name]
    if not bf.is_2d():
        raise ValueError("plot_surface is only for 2D functions (to visualize in 3D).")
    # get bounds and build evaluation grid
    lb, ub = bf.bounds
    X, Y, Z = surface_grid(bf.func, lb, ub, grid_points=grid)
    # prepare 3D figure and axes
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    # draw the surface with provided colormap and transparency
    ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=surf_alpha)
    # set view angle and labels
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"{fn_name} (3D surface)")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("f(x)")
    # if known, mark global minimum on the surface
    if bf.global_minimum is not None:
        gm = bf.global_minimum
        ax.scatter([gm[0]], [gm[1]], [bf.global_minimum_value], marker='X', s=80, c='gold', edgecolor='k')
    plt.tight_layout()
    plt.show()


def plot_contour(fn_name: str, grid=400, levels=60, cmap='viridis'):
    """Contour (filled) + contour lines and colorbar to inspect level sets."""
    # lookup benchmark function and validate dimensionality
    bf = registry[fn_name]
    if not bf.is_2d():
        raise ValueError("plot_contour is only for 2D functions.")
    # create grid and evaluate function
    lb, ub = bf.bounds
    X, Y, Z = surface_grid(bf.func, lb, ub, grid_points=grid)
    # create 2D axes and draw heatmap via pcolormesh
    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(X, Y, Z, shading='auto', cmap=cmap)
    # overlay contour lines for clearer level sets
    cs = ax.contour(X, Y, Z, levels=levels, colors='k', linewidths=0.6, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")
    # add colorbar and labels
    fig.colorbar(pcm, ax=ax, label='f(x)')
    ax.set_title(f"{fn_name} (contour & heatmap)")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    # mark global minimum if available
    if bf.global_minimum is not None:
        gm = bf.global_minimum
        ax.scatter([gm[0]], [gm[1]], color='red', marker='X', s=80, label='global min')
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_1d_slice(fn_name: str, fixed_coords: dict = None, axis=0, npoints=400):
    """Plot 1D slice of f along one coordinate while fixing others."""
    bf = registry[fn_name]
    # determine problem dimension from bounds
    lb, ub = bf.bounds
    d = lb.size
    if fixed_coords is None:
        fixed_coords = {}
    # start with center point of domain as template
    center = (lb + ub) / 2.0
    # override template coordinates with any user-specified fixed values
    for k, v in fixed_coords.items():
        assert 0 <= k < d, "fixed coordinate index out of range"
        center[k] = v

    # build array of x values along chosen axis and copy template for each
    xs = np.linspace(lb[axis], ub[axis], npoints)
    pts = np.tile(center, (npoints, 1))
    # vary only the selected axis across the slice
    pts[:, axis] = xs
    # evaluate the function along the slice and plot
    vals = bf.evaluate(pts)
    plt.figure(figsize=(8, 4))
    plt.plot(xs, vals, linewidth=1.8)
    plt.xlabel(f"x[{axis}]"); plt.ylabel("f(x)")
    plt.title(f"{fn_name} 1D slice (vary axis {axis})")
    plt.grid(True)
    plt.show()

def plot_surface_and_path(ax, X, Y, Z, path_points: np.ndarray = None, surf_alpha=0.7, cmap='viridis'):
    """Plot surface and overlay the path (path_points shape: (n_steps, 2))."""
    # draw surface on provided 3D axes
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, alpha=surf_alpha, cmap=cmap, rstride=3, cstride=3)

    if path_points is None:
        return

    # evaluate z-values at each path coordinate using interpolation on the grid
    zs = np.asarray([_interp_z(X, Y, Z, p[0], p[1]) for p in path_points])

    # add a small offset so markers/lines sit visibly above the surface
    zrange = np.nanmax(Z) - np.nanmin(Z)
    z_offset = 0.01 * (zrange if zrange != 0 else 1.0)
    zs_offset = zs + z_offset

    # plot the trajectory with strong styling for visibility
    ax.plot(path_points[:, 0], path_points[:, 1], zs_offset,
            linewidth=3.5, marker='o', markersize=7, label='best-so-far',
            zorder=20, color='red')

    # highlight the final (best) point
    ax.scatter([path_points[-1, 0]], [path_points[-1, 1]], [zs_offset[-1]],
               s=200, marker='X', edgecolor='black', linewidth=1.5, zorder=30, c='gold')

    # annotate start and end points for clarity
    ax.text(path_points[0, 0], path_points[0, 1], zs_offset[0],
            "start", color='black', zorder=40)
    ax.text(path_points[-1, 0], path_points[-1, 1], zs_offset[-1],
            "best", color='black', zorder=40)

def _interp_z(X, Y, Z, xq, yq):
    """Simple bilinear interpolation on grid to get z(xq,yq)."""
    # find indices of grid cell containing query point
    xi = np.searchsorted(X[0, :], xq) - 1
    yi = np.searchsorted(Y[:, 0], yq) - 1
    # clamp to valid index range so we can access neighbors safely
    xi = np.clip(xi, 0, X.shape[1]-2)
    yi = np.clip(yi, 0, Y.shape[0]-2)

    # corner coordinates of the cell
    x1, x2 = X[0, xi], X[0, xi+1]
    y1, y2 = Y[yi, 0], Y[yi+1, 0]
    Q11 = Z[yi, xi]; Q21 = Z[yi, xi+1]; Q12 = Z[yi+1, xi]; Q22 = Z[yi+1, xi+1]

    # perform bilinear interpolation; handle degenerate cell if denom == 0
    denom = (x2 - x1) * (y2 - y1)
    if denom == 0:
        return Q11
    wx2 = (xq - x1) / (x2 - x1)
    wy2 = (yq - y1) / (y2 - y1)
    return (Q11 * (1-wx2) * (1-wy2) + Q21 * wx2 * (1-wy2) +
            Q12 * (1-wx2) * wy2 + Q22 * wx2 * wy2)

def plot_convergence(ax, best_f_history):
    # plot best-so-far objective history on a log y-scale for convergence behavior
    ax.plot(best_f_history, linewidth=2.0, color='tab:orange')
    ax.set_yscale('log')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best f (log scale)')
    ax.grid(True)

def plot_neighbors_on_surface(ax, X, Y, Z, neighbors: np.ndarray, marker='.', markersize=5, alpha=0.6, color='black'):
    """Plot neighbors (shape (n,d)) as small scatter points slightly above the surface, centers the marker z-values using bilinear interpolation of Z at (x,y) and adds offset."""
    # nothing to do if no neighbors provided
    if neighbors is None or neighbors.size == 0:
        return

    # ensure neighbors have expected 2D shape for visualization
    if neighbors.ndim != 2 or neighbors.shape[1] != 2:
        raise ValueError("neighbors must be shape (n,2) for 2D visualization")

    # compute z coordinates for each neighbor by interpolating on the surface grid
    zs = np.asarray([_interp_z(X, Y, Z, p[0], p[1]) for p in neighbors])
    # offset points slightly above the surface based on global z-range for visibility
    zrange = np.nanmax(Z) - np.nanmin(Z)
    offset = 0.02 * (zrange if zrange != 0 else 1.0)
    zs_offset = zs + offset

    # draw neighbor points as small markers with given styling
    ax.scatter(neighbors[:, 0], neighbors[:, 1], zs_offset, s=markersize, marker=marker, alpha=alpha, color=color, zorder=15)