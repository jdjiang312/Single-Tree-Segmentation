"""
Statement:

    This code is not our work,  it's our way of reproducing the authors' work by trying to understand their methods.

    The information of this work is below:

    Article: Individual tree crown segmentation from airborne LiDAR data using a novel Gaussian filter and energy function minimization-based approach
    Authors: Ting Yun, Kang Jiang, Guangchao Li, et al.
    Citation: Yun T, Jiang K, Li G, et al. Individual tree crown segmentation from airborne LiDAR data using a novel Gaussian filter and energy function minimization-based approach[J]. Remote Sensing of Environment, 2021, 256: 112307.

"""


import numpy as np
import laspy
from scipy.ndimage import grey_dilation, grey_erosion
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed as skwatershed
from scipy.ndimage import sobel

# 读取点云数据(.las .laz .xyz .txt)
def read_point_cloud(filename):
    if filename.endswith('.las') or filename.endswith('.laz'):
        with laspy.open(filename) as file:
            las = file.read()
            points = np.vstack((las.x, las.y, las.z)).transpose()
    elif filename.endswith('.xyz') or filename.endswith('.txt'):
        points = np.loadtxt(filename)
    else:
        raise ValueError("Unsupported file format")
    return points


# 点云去噪,kNN μ-σ< valid points <μ+σ
def denoise_point_cloud(points, k=30):
    tree = cKDTree(points)
    # 查询每个点的k个最近邻居（包含自身，所以实际查找k+1个）
    distances, indices = tree.query(points, k=k + 1)

    # 计算k个邻居的均值和方差，排除自身所以从1开始
    mean_vals = np.mean(points[indices[:, 1:], :], axis=1)
    std_vals = np.std(points[indices[:, 1:], :], axis=1)

    # 检查每个点是否在其均值±方差的范围内
    denoised_points = []
    for i, point in enumerate(points):
        if np.all(point >= mean_vals[i] - std_vals[i]) and np.all(point <= mean_vals[i] + std_vals[i]):
            denoised_points.append(point)

    return np.array(denoised_points)


# 构建DSM
def create_dsm(points, grid_size):
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)
    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max, grid_size),
        np.arange(y_min, y_max, grid_size)
    )
    # dsm = np.zeros(grid_x.shape)  # 将dsm初始值设为0
    dsm = np.full(grid_x.shape, np.nan)

    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            x1, x2 = grid_x[i, j], grid_x[i, j] + grid_size
            y1, y2 = grid_y[i, j], grid_y[i, j] + grid_size
            mask = (points[:, 0] >= x1) & (points[:, 0] <= x2) & (points[:, 1] >= y1) & (points[:, 1] <= y2)
            if np.any(mask):
                dsm[i, j] = np.max(points[mask, 2])
    return dsm


# 图形学处理:膨胀+腐蚀
def morphological_opening(dsm, size1, size2):
    dilated = grey_dilation(dsm, size=(size1, size1))
    eroded = grey_erosion(dilated, size=(size2, size2))
    return eroded


# 双高斯滤波
def dual_gaussian_filter(dsm, s, a1, a2):
    filtered_dsm = np.copy(dsm)
    height, width = dsm.shape
    for i in range(height):
        for j in range(width):
            if np.isnan(dsm[i, j]):
                continue
            sum_weights = 0
            sum_values = 0
            for di in range(-s, s + 1):
                for dj in range(-s, s + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width and not np.isnan(dsm[ni, nj]):
                        distance = np.sqrt(di ** 2 + dj ** 2)
                        height_diff = dsm[ni, nj] - dsm[i, j]
                        sigma_d = a1 * dsm[i, j] if dsm[i, j] != 0 else 1e-10  # 避免除以零
                        sigma_g = sigma_d / a2 if a2 != 0 else 1e-10  # 同上
                        g1 = np.exp(-distance ** 2 / (2 * sigma_d ** 2))
                        g2 = np.exp(-height_diff ** 2 / (2 * sigma_g ** 2))
                        weight = g1 + g2
                        sum_weights += weight
                        sum_values += weight * dsm[ni, nj]
            filtered_dsm[i, j] = sum_values / sum_weights if sum_weights != 0 else dsm[i, j]
    return filtered_dsm


def visualize_dsm(original_dsm, filtered_dsm):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_dsm, cmap='terrain')
    plt.title('Original DSM')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_dsm, cmap='terrain')
    plt.title('Filtered DSM')
    plt.colorbar()

    plt.show()


############################### treetop detection #####################################
'''
treetop detection:
step1：ISODATA聚类，得到n个类
step2：计算n个类别的平均值
step3：计算τ=mean(每个聚类中心的均值)
step4：局部最大值检测，满足两个条件：1. 在半径r范围内满足局部高程最大；2. 在半径r的领域范围内所有grid格网的高程平均值应大于τ×α，其中α是人为设定的参数
（α在原论文中没有详细介绍参数设置建议，通过测试得到alpha=0.8较为合适）

Reference: 
Faraji M, Shanbehzadeh J, Nasrollahi K, et al. Extremal regions detection guided by maxima of gradient magnitude[J]. IEEE Transactions on Image Processing, 2015, 24(12): 5401-5415.

##### 原论文直接在matlab中调用函数imregionalmax，将该函数的源代码转写成python效果极差，代码如下：
def treetop_detection(dsm,conn=8):
    if conn == 4:
        # 4连接性对应于上下左右四个方向
        struct = generate_binary_structure(2, 1)
    else:
        # 8连接性对应于上下左右以及对角四个方向
        struct = generate_binary_structure(2, 2)
    local_max = maximum_filter(dsm, footprint=struct, mode='constant', cval=np.min(dsm) - 1)
    detected_treetops = (dsm == local_max)
    # 获取局部最大值的坐标
    yx_coords = np.where(detected_treetops)
    # 提取对应的高程值
    elevations = dsm[yx_coords]

    # 组合坐标和高程值
    detected_treetops = np.array(list(zip(yx_coords[0], yx_coords[1], elevations)))
    return detected_treetops

treetops = treetop_detection(filtered_dsm, conn=8)      # 示例使用时使用这行代码
'''

# 1. ISODATA聚类
def isodata_clustering(dsm, n_clusters=14, max_iter=300, tol=1e-4, split_criteria=0.5, min_cluster_size=10,
                       max_clusters=20, merge_criteria=1.5):
    # 初始有效点和K-Means聚类
    valid_points = dsm[~np.isnan(dsm)].reshape(-1, 1)
    clusters = KMeans(n_clusters=n_clusters, max_iter=max_iter, tol=tol, n_init=10).fit(valid_points)
    labels = clusters.labels_
    centers = clusters.cluster_centers_

    for iteration in range(max_iter):
        # 计算每个簇的标准差
        std_devs = np.array([valid_points[labels == i].std() for i in range(n_clusters)])

        # 拆分标准：标准差大且簇内点数足够
        large_std_dev_indices = np.where((std_devs > split_criteria) & (np.bincount(labels) > 2 * min_cluster_size))[0]
        if len(large_std_dev_indices) > 0 and n_clusters < max_clusters:
            for index in large_std_dev_indices:
                centers = np.append(centers, [centers[index] + tol], axis=0)  # 拆分簇中心
                centers[index] -= tol
            n_clusters += len(large_std_dev_indices)
            clusters = KMeans(n_clusters=n_clusters, init=centers, n_init=1, max_iter=max_iter, tol=tol).fit(
                valid_points)
            labels = clusters.labels_
            centers = clusters.cluster_centers_

        # 合并标准：簇中心距离小于合并标准
        center_distances = cdist(centers, centers)
        np.fill_diagonal(center_distances, np.inf)
        to_merge = np.where(center_distances < merge_criteria)
        unique_to_merge = np.unique(to_merge[0])
        if len(unique_to_merge) > 1:
            centers = np.delete(centers, unique_to_merge[1:], 0)
            n_clusters -= len(unique_to_merge) - 1
            clusters = KMeans(n_clusters=n_clusters, init=centers, n_init=1, max_iter=max_iter, tol=tol).fit(
                valid_points)
            labels = clusters.labels_
            centers = clusters.cluster_centers_

        if len(large_std_dev_indices) == 0 and len(unique_to_merge) <= 1:
            break  # 没有进一步的拆分或合并，结束循环

    # 构建标签映射
    label_map = np.full(dsm.shape, -1)  # 使用-1表示NaN区域
    label_map[~np.isnan(dsm)] = labels
    return label_map, centers


# 2. 计算参数tau
def calculate_tau(dsm, label_map, cluster_centers):
    cluster_averages = []
    for i, center in enumerate(cluster_centers):
        cluster_points = dsm[label_map == i]
        cluster_average = np.nanmean(cluster_points)
        cluster_averages.append(cluster_average)
    tau = np.mean(cluster_averages)
    return tau


# 3. 执行treetop detection
def treetop_detection(dsm, tau, alpha, r):
    detected_treetops = []
    height, width = dsm.shape
    for i in range(height):
        for j in range(width):
            if np.isnan(dsm[i, j]):
                continue
            min_i, max_i = max(0, i - r), min(height, i + r + 1)
            min_j, max_j = max(0, j - r), min(width, j + r + 1)
            neighbourhood = dsm[min_i:max_i, min_j:max_j]
            if dsm[i, j] != np.nanmax(neighbourhood) or np.nanmean(neighbourhood) <= tau * alpha:
                continue
            detected_treetops.append((i, j))

##### 下面步骤是为了后续treetop screening时按照treetops之间的最短距离来构建band #####
    # 以距离坐标原点距离最短的点为first point
    first_point = min(detected_treetops, key=lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2))
    ordered_treetops = [first_point]

    # 从detected_treetops中移除first_point
    detected_treetops.remove(first_point)

    # 根据距离最近的treetop点进行排序，这一步方便后续对treetops进行chain连接
    while detected_treetops:
        closest_point = min(detected_treetops, key=lambda x: np.sqrt((x[0] - ordered_treetops[-1][0]) ** 2 + (x[1] - ordered_treetops[-1][1]) ** 2))
        ordered_treetops.append(closest_point)
        detected_treetops.remove(closest_point)

    detected_treetops = ordered_treetops
    return detected_treetops


# 可视化树冠检测结果
def visualize_treetops(filtered_dsm, treetops, marker_color='red'):
    plt.figure(figsize=(10, 8))
    plt.imshow(filtered_dsm, cmap='terrain')
    plt.colorbar(label='Elevation')
    for treetop in treetops:
        plt.plot(treetop[1], treetop[0], marker='o', color=marker_color, markersize=5)
    plt.title('Treetop Detection')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# false treetop screening
def false_treetop_screening(treetops, filtered_dsm, theta_threshold):
    # 将filtered_dsm的nan设为0
    filtered_dsm = np.nan_to_num(filtered_dsm, nan=0)

    # 计算角度函数
    def calculate_angle(p1, p2, p3):
        vector1 = np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])
        vector2 = np.array([p2[0] - p3[0], p2[1] - p3[1], p2[2] - p3[2]])
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        cos_angle = dot_product / (norm1 * norm2)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    # 计算树顶点的z坐标
    treetops_xyz = [[x, y, filtered_dsm[int(x), int(y)]] for x, y in treetops]

    # 构建相邻两个treetop之间3个像素宽度的band neighborhood
    def construct_bands(treetops_xyz, filtered_dsm):
        bands = []
        for i in range(len(treetops_xyz) - 1):
            p1, p2 = treetops_xyz[i], treetops_xyz[i + 1]
            line_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            norm_line_vec = np.linalg.norm(line_vec)
            if norm_line_vec == 0:
                continue
            perp_vec = np.array([-line_vec[1], line_vec[0]]) / norm_line_vec
            band_points = []
            for x in range(filtered_dsm.shape[0]):
                for y in range(filtered_dsm.shape[1]):
                    point_vec = np.array([x - p1[0], y - p1[1]])
                    proj_length = np.dot(point_vec, line_vec) / norm_line_vec
                    dist_to_line = np.linalg.norm(point_vec - proj_length * (line_vec / norm_line_vec))
                    if 0 <= proj_length <= norm_line_vec and dist_to_line <= 2:
                        z_value = filtered_dsm[x, y]
                        if not np.isnan(z_value):
                            band_points.append([x, y, z_value])
            bands.append(band_points)
        return bands

    bands = construct_bands(treetops_xyz, filtered_dsm)

    min_value_grids = []
    for band in bands:
        if band:
            min_value_grids.append(min(band, key=lambda x: x[2]))
        else:
            p1_x, p1_y = treetops_xyz[i][:2]
            p2_x, p2_y = treetops_xyz[i + 1][:2]
            mid_x = (p1_x + p2_x) / 2
            mid_y = (p1_y + p2_y) / 2
            min_value_grids.append([mid_x, mid_y, 0])

    retained_treetops = []
    removed_treetops = []
    for i, min_grid in enumerate(min_value_grids):
        p1 = treetops_xyz[i]
        p2 = min_grid
        p3 = treetops_xyz[(i + 1) % len(treetops_xyz)]

        theta = calculate_angle(p1, p2, p3)
        if np.isnan(theta):
            theta = 0
        if theta > theta_threshold:
            if p1[2] > p3[2]:
                retained_treetops.append(p1[:2])
                removed_treetops.append(p3[:2])
            else:
                retained_treetops.append(p3[:2])
                removed_treetops.append(p1[:2])
        else:
            retained_treetops.append(p1[:2])
            retained_treetops.append(p3[:2])

    retained_treetops = list(set(tuple(tp) for tp in retained_treetops))
    removed_treetops = list(set(tuple(tp) for tp in removed_treetops))

    # 移除在removed_treetops中的点
    retained_treetops = [tp for tp in retained_treetops if tp not in removed_treetops]

    # 赋予retained_treetops高程信息
    retained_treetops_with_elevation = [[x, y, filtered_dsm[int(x), int(y)]] for x, y in retained_treetops]

    retained_treetops = retained_treetops_with_elevation

    return retained_treetops, removed_treetops, treetops_xyz, min_value_grids, bands


def visualize_screening_results_with_bands(filtered_dsm, retained_treetops, removed_treetops, min_value_grids, bands):
    plt.figure(figsize=(10, 8))
    plt.imshow(filtered_dsm, cmap='terrain')
    plt.colorbar(label='Elevation')

    # 绘制保留的treetop点
    for treetop in retained_treetops:
        plt.plot(treetop[1], treetop[0], 's', color='red')  # 使用红色方块表示true treetops

    # 绘制移除的treetop点
    for treetop in removed_treetops:
        plt.plot(treetop[1], treetop[0], 'rx')  # 使用红色叉号表示false treetops

    # 绘制条带中的min_value_grids点
    for min_grid in min_value_grids:
        plt.plot(min_grid[1], min_grid[0], 's', color='yellow')  # 使用黄色方块表示min_value_girds

    # 绘制每个条带内的DSM格网点
    for band in bands:
        for point in band:
            plt.plot(point[1], point[0], 's', color='orange', alpha=0.03)  # 使用半透明橘色方块表示bands neighborhood

    plt.title('Treetop Screening Results')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


############################### 分水岭算法检测树冠边界 #####################################
# 高斯滤波处理filtered_dsm,为了去除crown中的nan
def apply_gaussian_filter(dsm, sigma=1):

    # 过滤NaN值，将它们替换为0，因为高斯滤波不能直接应用于NaN
    dsm_no_nan = np.nan_to_num(dsm, copy=True)

    # 高斯滤波
    gaussian_dsm = gaussian_filter(dsm_no_nan, sigma=sigma)

    # 将原来为NaN的位置重新设置为NaN
    gaussian_dsm[np.isnan(dsm)] = np.nan

    return gaussian_dsm


# 定义利用sobel算子计算dsm梯度函数
def compute_gradient(dsm):
    dx = sobel(dsm, axis=0, mode='constant')
    dy = sobel(dsm, axis=1, mode='constant')
    grad_dsm = np.hypot(dx, dy)
    return grad_dsm


# 计算可视化梯度图像函数
def visualize_gradient(grad_dsm):
    plt.figure(figsize=(8, 6))
    plt.imshow(grad_dsm, cmap='viridis')
    plt.title('Gradient of DSM')
    plt.colorbar()
    plt.show()


# 定义拓展dsm边界函数（这一步是为了防止位于dsm图像边缘的树冠边界线无法识别的问题）
def expand_dsm_boundary(dsm, border_size=1):
    """
    通过在图像周围添加一个边框（恒定值为 0）来扩展 DSM 边界
    """
    expanded_dsm = np.pad(dsm, pad_width=border_size, mode='constant', constant_values=0)
    return expanded_dsm


# 更新坐标函数（如果仅仅扩展边界会导致treetops偏离dsm，因此需要更新retained_treetops坐标）
def update_treetop_coordinates(treetops, border_size=1):
    return [(x + border_size, y + border_size, z) for x, y, z in treetops]


# 分水岭算法执行函数
def watershed_segmentation(gradient_dsm, retained_treetops, dsm):
    # 创建一个空的图像用于存放标记，尺寸与dsm相同
    markers = np.zeros_like(dsm, dtype=int)

    # 对每个保留的treetop点进行标记
    for i, (x, y, _) in enumerate(retained_treetops, start=1):
        markers[int(x), int(y)] = i

    # 使用分水岭算法进行分割
    labels = skwatershed(gradient_dsm, markers, mask=~np.isnan(dsm))

    return labels


# 可视化分割边界线的函数
def visualize_watershed_boundaries(dsm, labels, retained_treetops):
    plt.figure(figsize=(10, 8))
    plt.imshow(dsm, cmap='terrain')
    plt.colorbar(label='Elevation')

    for treetop in retained_treetops:
        plt.plot(treetop[1], treetop[0], 's', color='red')  # 使用红色方块表示true treetops

    # 寻找并绘制边界
    boundaries = find_boundaries(labels, mode='thick')
    plt.contour(boundaries, [0.5], colors='red')

    plt.title('Watershed Segmentation Boundaries')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()



############################### 能量方程控制的水膨胀算法 #####################################
# 根据treetop划分区间，并构建同心轮廓
# def split_dsm_by_height(filtered_dsm, treetops, height_interval=3.5, levels=5):
#     contour_maps = []
#     for treetop in treetops:
#         contours = []
#         for level in range(levels):
#             height_threshold = filtered_dsm[treetop[0], treetop[1]] - (level * height_interval)
#             mask = (filtered_dsm >= height_threshold) & (filtered_dsm < height_threshold + height_interval)
#             contours.append(mask)
#         contour_maps.append(contours)
#     return contour_maps
#
#
# # 计算4邻域梯度信息，对边界单元格进行0填充
# def calculate_gradient_dsm(filtered_dsm):
#     # 填充数组
#     padded_dsm = np.pad(filtered_dsm, ((1, 1), (1, 1)), 'constant', constant_values=0)
#
#     # 计算x方向和y方向的梯度
#     grad_x = np.zeros_like(filtered_dsm)
#     grad_y = np.zeros_like(filtered_dsm)
#
#     # 用中心差分计算内部像素的梯度
#     grad_x[:, :] = (padded_dsm[1:-1, 2:] - padded_dsm[1:-1, :-2]) / 2
#     grad_y[:, :] = (padded_dsm[2:, 1:-1] - padded_dsm[:-2, 1:-1]) / 2
#
#     return grad_x, grad_y
#
#
# # 构建Delaunay三角形
# def build_delaunay_triangulation(treetops):
#     points = np.array(treetops)
#     tri = Delaunay(points)
#     return tri
#
#
# def extract_gradient_dsm_in_triangles_and_levels(tri, contour_maps, gradient_dsm):
#     triangle_gradients = []
#     for simplex in tri.simplices:
#         # 获取三角形顶点坐标
#         vertices = tri.points[simplex]
#         # 创建三角形路径
#         triangle_path = mpath.Path(vertices)
#
#         triangle_contours = []
#         for level_contours in contour_maps:
#             # 对每个高度区间进行处理
#             level_gradient = []
#             for contour in level_contours:
#                 # 检查哪些点在当前高度区间和三角形内
#                 y_idxs, x_idxs = np.where(contour)
#                 for y, x in zip(y_idxs, x_idxs):
#                     if triangle_path.contains_points([(x, y)]):
#                         # 提取gradient_DSM中的数据
#                         level_gradient.append(gradient_dsm[y, x])
#             triangle_contours.append(np.array(level_gradient))
#         triangle_gradients.append(triangle_contours)
#     return triangle_gradients
#
#
# # 初始化水体扩张状态
# def init_water_expansion_status(filtered_dsm):
#     return np.zeros_like(filtered_dsm, dtype=bool)
#
#
# # 计算两个向量之间的角度
# def angle_between(v1, v2):
#     v1_u = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) != 0 else v1
#     v2_u = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) != 0 else v2
#     angle = 180 - np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
#     return angle
#
#
# def calculate_energy(cell_index, triangle_dsm, dsm, treetop_index, triangulation, alpha):
#     treetop_location = triangulation.points[treetop_index]
#
#     # 寻找树顶点的邻居节点（Delaunay三角剖分的其他两个节点）
#     neighbors_indices = triangulation.vertex_neighbor_vertices[1][
#                         triangulation.vertex_neighbor_vertices[0][treetop_index]:
#                         triangulation.vertex_neighbor_vertices[0][treetop_index + 1]
#                         ]
#     treetop_neighbours = triangulation.points[neighbors_indices]
#
#     # z_c_d_b 是边界单元的平均高度值
#     # 假设是以 treetop 为中心，在一定高度范围内的所有点的平均高度
#     z_c_d_b = np.mean([dsm[c[0], c[1]] for c in treetop_neighbours if not np.isnan(dsm[c[0], c[1]])])
#
#     # Q_b 是在第 b 个同心圆轮廓上的边界单元数量
#     # 假设同心轮廓是通过树顶点划分的高度区间的轮廓
#     Q_b = len(treetop_neighbours)
#
#     # 计算 beta 值，取决于 z_c_d_b 相对于 treetop 的高度值
#     beta = 1 if dsm[treetop_location[0], treetop_location[1]] - z_c_d_b >= 0 else -1
#
#     # 计算能量函数
#     energy = 0
#     for neighbor_location in treetop_neighbours:
#         # 计算边界单元的梯度向量
#         gradient_vector = triangle_dsm[neighbor_location[0], neighbor_location[1]]
#
#         # 计算 treetop 到边界单元的向量
#         vector_treetop_to_cell = neighbor_location - treetop_location
#
#         # 计算向量之间的角度
#         angle = angle_between(vector_treetop_to_cell, gradient_vector)
#
#         # 计算第一部分能量值
#         energy += (alpha / Q_b) * np.cos(angle)
#
#         # 计算第二部分能量值
#         energy += beta / 3 * (z_c_d_b - dsm[neighbor_location[0], neighbor_location[1]])
#
#     return energy



# 执行代码：

# 1. 输入文件路径
filename = 'F:/TreeSeparation-master/reference/RSE_Yun.T/test6.xyz'

# 2. 读取point cloud
points = read_point_cloud(filename)

# 3. 点云去噪
denoised_points = denoise_point_cloud(points)

# 4. 构建DSM
dsm = create_dsm(denoised_points, grid_size=0.18)

# 5. 图形学操作（膨胀和腐蚀）
opened_dsm = morphological_opening(dsm, size1=5, size2=3)

# 6. 双高斯滤波
filtered_dsm = dual_gaussian_filter(opened_dsm, s=3, a1=0.3, a2=2.0)  ## s适当改大可以避免false treetop的检测
# visualize_dsm(opened_dsm, filtered_dsm)

# 7. ISODATA
label_map, centers = isodata_clustering(filtered_dsm)

# 8. 计算tau
tau = calculate_tau(filtered_dsm, label_map, centers)

# 9. treetop detection
treetops = treetop_detection(filtered_dsm, tau, alpha=0.8, r=7)  ## r适当改大可以避免false treetop的检测

# 10. 可视化检测到的treetops点
# visualize_treetops(filtered_dsm, treetops, marker_color='red')

# 11，false treetops descreening
retained_treetops, removed_treetops, treetops_chained, min_value_grids, bands = false_treetop_screening(treetops, filtered_dsm, theta_threshold=165)

# 12. 可视化true treetops和false treetops
visualize_screening_results_with_bands(filtered_dsm, retained_treetops, removed_treetops, min_value_grids,  bands)

# 13. 分水岭算法检测树冠边界线
filtered_dsm = np.nan_to_num(filtered_dsm, nan=0)
gaussian_dsm = apply_gaussian_filter(filtered_dsm, sigma=1.7)
grad_dsm = compute_gradient(gaussian_dsm)
grad_dsm = expand_dsm_boundary(grad_dsm, border_size=10)  # 拓展边界
gaussian_dsm = expand_dsm_boundary(gaussian_dsm, border_size=10)  # 拓展边界
retained_treetops = update_treetop_coordinates(retained_treetops, border_size=10)  # 更新retained_treetops坐标
# visualize_gradient(grad_dsm)
labels = watershed_segmentation(grad_dsm, retained_treetops, grad_dsm)
visualize_watershed_boundaries(grad_dsm, labels, retained_treetops)




