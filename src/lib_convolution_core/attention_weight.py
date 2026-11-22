from typing import Callable, Literal
import functools
import scipy.stats

import numpy

from src.lib_common_math.common_math import point_distance_norm

def central_peripheral_weight__norm__normal_distribution(
    central_pixel_attributes: numpy.ndarray,
    peripheral_pixel_attributes: numpy.ndarray,
    norm_type: Literal["fro", "nuc"] | int | float | None = None,
    mean: float = 0.0,
    standard_deviation: float = 1.0,
    axis_dimension_index: int | None = -1,
) -> numpy.ndarray:
    """
    约定：
    1. 中心像素属性为 central_pixel_attributes
    2. 周边像素属性为 peripheral_pixel_attributes
    3. 中心像素属性和周边像素属性的维度相同
    4. 以上属性的前两位为坐标，后续为其他属性

    计算中心像素属性和周边像素属性 的距离（范数可调），
    并将其作为正态分布的输入，返回正态分布的概率密度函数值
    :param central_pixel_attributes: 中心像素属性(必须包含坐标)
    :param peripheral_pixel_attributes: 周边像素属性(必须包含坐标)
    :param norm_type: 范数类型，默认为 2 范数
    :param mean: 正态分布的均值，默认为 0
    :param standard_deviation: 正态分布的标准差，默认为 1
    :param axis_dimension_index: 计算范数的轴，默认为最后一个轴
    :return: 正态分布的概率密度函数值
    """
    # ============================== 距离 ==============================
    # 计算中心像素属性和周边像素属性在指定轴上的差值
    center_peripheral_norm_value: numpy.ndarray = point_distance_norm(
        central_pixel_attributes,
        peripheral_pixel_attributes,
        norm_type=norm_type,
        axis_dimension_index=axis_dimension_index
    )


    # ============================== 正态分布 ==============================
    # 计算正态分布的概率密度函数值
    normal_distribution_value: numpy.ndarray = numpy.asarray(
        scipy.stats.norm.pdf(
            x=center_peripheral_norm_value,
            loc=mean,
            scale=standard_deviation,
        )
    )

    # 去掉坐标那个维度，返回概率密度函数值
    normal_distribution_value = numpy.squeeze(normal_distribution_value, axis=axis_dimension_index)

    return normal_distribution_value


def generate_distribution_kernel(
    kernel_size: int,
    distribution_function: Callable[
        [
            numpy.ndarray,
            numpy.ndarray,
        ],
        numpy.ndarray
    ] = functools.partial(
            central_peripheral_weight__norm__normal_distribution,
            norm_type=2,
            mean=0,
            standard_deviation=1,
            axis_dimension_index=-1,
        ),
    axis_dimension_index=-1,
    center_coordinate_on_distribution_function: numpy.ndarray = numpy.array([0, 0]),
    dtype: numpy.typing.DTypeLike = numpy.float64
) -> numpy.ndarray:
    """
    生成一个符合正态分布的卷积核

    :param kernel_size: 卷积核的大小
    :param distribution_function: 分布函数，默认是正态分布，返回概率密度函数值
    :param center_coordinate_on_distribution_function: 中心点坐标，默认是 (0, 0)
    :param axis_dimension_index: 计算范数的轴，默认为最后一个轴
    :param dtype: 数据类型，默认是 numpy.float64
    :return: 卷积核矩阵，形状为 (kernel_size, kernel_size)
    """
    matrix_center_coordinate: float = (kernel_size - 1) / 2.0
    matrix_center_coordinate_lefted: int = int(matrix_center_coordinate)
    # =============================== 生成一维坐标 ===============================
    coordinates: numpy.ndarray = numpy.arange(
        kernel_size,
        dtype=dtype
    ) - matrix_center_coordinate

    # =============================== 生成 ===============================
    # meshgrid 意思是沿着 各自的 方向扩展
    # 比如
    # coordinates = [-1, 0, 1]
    # meshgrid(coordinates, coordinates, indexing="ij") 会生成
    # coordinate_x = [[-1, -1, -1],
    #                 [ 0,  0,  0],
    #                 [ 1,  1,  1]]
    # coordinate_y = [[-1,  0,  1],
    #                 [-1,  0,  1],
    #                 [-1,  0,  1]]
    # 以 0,0 为中心，生成一个二维坐标矩阵
    coordinate_x_expand: numpy.ndarray
    coordinate_y_expand: numpy.ndarray
    coordinate_x_expand, coordinate_y_expand = numpy.meshgrid(
        coordinates, coordinates, indexing="ij"
    )

    # 在最后一个维度上堆叠坐标矩阵，形成一个形状为 (kernel_size, kernel_size, 2) 的网格坐标矩阵
    # stack 在第N个维度上堆叠矩阵
    coordinate_x_y_grid: numpy.ndarray = numpy.stack(
        [coordinate_x_expand, coordinate_y_expand],
        axis=axis_dimension_index
    )  # (kernel_size, kernel_size, 2)

    # ============================== 计算卷积核 ==============================
    # 计算正态分布的概率密度函数值
    kernel: numpy.ndarray = distribution_function(
        center_coordinate_on_distribution_function,  # 中心点坐标
        coordinate_x_y_grid,  # 周边点坐标,
    )

    # ============================== 调整中心值 ==============================
    # 中心是一个大负数，为所有东西（中心除外）的和
    # 求和
    sum_of_kernel: numpy.ndarray = numpy.sum(kernel)
    # 1 归一化 差值
    # < 0
    difference_to_one: numpy.ndarray = 1 - sum_of_kernel

    # 调整中心值，使得卷积核的和为 1
    if kernel_size % 2 == 1:
        kernel[matrix_center_coordinate_lefted, matrix_center_coordinate_lefted] += difference_to_one
    else:
        kernel[
            matrix_center_coordinate_lefted: matrix_center_coordinate_lefted + 2,
            matrix_center_coordinate_lefted: matrix_center_coordinate_lefted + 2
        ] += difference_to_one / 4  # 平均分配到四个中心点

    # 调整中心值，使得卷积核的和为 1
    # output: [kernel_size, kernel_size]
    return kernel
