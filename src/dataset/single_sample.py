import numpy as np

# your data (GT mesh as point cloud [N, 3])
gt_mesh = np.array([
    -1.8233562, -3.2887428, -3.3480015, -1.8190439, -3.14004, -3.4705594, -1.8404241, -3.8773122, -2.8629148, -1.1154985, -3.063245, -1.4943037, -0.90766495, -2.8298547, -1.1019278, -0.5100329, -2.929736, -1.2742212, -1.7557249, -0.95654994, -5.2701473, -1.2683773, -0.4092748, -4.3500676, -0.7779664, -0.5324609, -4.562562, -3.7392468, 1.6591635, 6.506395, -3.7563672, 4.8768063, 4.6015687, 3.4655972, -3.648517, -3.4057674, 3.7394013, -1.6882095, -6.4891996, 3.7160668, -1.6613152, -6.509821, 3.7238894, -1.6703311, -6.502908, 3.7392468, -1.6591635, -6.506395, 2.3462203, -3.647197, -2.511832, 3.4278336, -3.918887, -2.9804935, 1.3526479, 2.1318183, 7.2590394, 1.3355273, 5.3494606, 5.354213
], dtype=np.float16).reshape(-1, 3)

# existing mesh as point cloud [N, 3] (empty initial state)
cur_mesh = np.empty((0, 3), dtype=np.float16)

# empty history (2D params with 10 columns, and empty token list)
hist_params = np.empty((0, 10), dtype=np.float16)
hist_tokens = np.empty((0,), dtype=np.int8)

# next params: N x 10   (last entry = {0 neg, 1 pos})
next_params = np.array([
    [3.7392354, 13.984732, 5.1688747, 0, 0, 0, 327.87152, 80.09921, 271.5259, 1],
    [11.95524, 13.526094, 10.227897, 4.9055624, 7.085265, 8.182921, 124.38747, 335.67645, 264.4143, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=np.float16)

# next tokens (same length as rows in next_params)
next_tokens = np.array([0, 0, 4], dtype=np.int8)

# Save a single example (compressed, with smaller dtypes on disk)
np.savez_compressed(
    "sample.npz",
    gt_mesh=gt_mesh.astype(np.float16),
    cur_mesh=cur_mesh.astype(np.float16),
    hist_params=hist_params.astype(np.float16),
    hist_tokens=hist_tokens.astype(np.int8),
    next_params=next_params.astype(np.float16),
    next_tokens=next_tokens.astype(np.int8),
)