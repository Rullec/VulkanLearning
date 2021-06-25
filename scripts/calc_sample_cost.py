N = 10
noised_samples = 12
avg_cost = 15  # unit: s

num_of_prop = int(N * N * (N + 1) / 2)
num_of_meshes = num_of_prop * 12
sample_cost_time_seconds = num_of_meshes * avg_cost
sample_cost_time_days = sample_cost_time_seconds / (3600 * 24)

num_of_coupled_views = 2
num_of_init_rot = 4
num_of_noised_cam_pos = 4
num_of_images = num_of_meshes * num_of_noised_cam_pos * num_of_init_rot * num_of_coupled_views
num_of_pixels = num_of_images * 300 * 300
num_of_bytes_per_pixel = 4
num_of_bytes = num_of_pixels * 4
num_of_float_GB = num_of_bytes / (1024 * 1024 * 1024)
num_of_byte_kb = num_of_images * 10
num_of_byte_gb = num_of_byte_kb / 1e6

print(f"N = {N}")
print(f"num of prop = {num_of_prop}")
print(f"num of meshes = {num_of_meshes}")
print(f"cost {sample_cost_time_days} day")
print(f"num of images {num_of_images}")
print(f"num of GB(disk) {num_of_byte_gb}")
print(f"num of GB(float) {num_of_float_GB}")