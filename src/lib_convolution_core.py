from typing import Callable, cast, Literal, Any
import functools
import scipy.stats

import numpy
import torch


def central_peripheral_weight__norm__normal_distribution(
    central_pixel_attributes: numpy.ndarray,
    peripheral_pixel_attributes: numpy.ndarray,
    norm_type: Literal["fro", "nuc"] | int | float | None = None,
    mean: float = 0.0,
    standard_deviation: float = 1.0,
    axis: int | None = -1
) -> numpy.ndarray:
    """
    约定：
    1. 中心像素属性为 central_pixel_attributes
    2. 周边像素属性为 peripheral_pixel_attributes
    3. 中心像素属性和周边像素属性的维度相同
    4. 以上属性的前两位为坐标，后续为其他属性
    """
    # 计算坐标的差
    center_peripheral_axis_difference: numpy.ndarray | torch.Tensor =\
        central_pixel_attributes- peripheral_pixel_attributes

    # print("center shape:", central_pixel_attributes.shape)
    # print("peripheral shape:", peripheral_pixel_attributes.shape)
    # 计算坐标的范数
    center_peripheral_norm_value: numpy.float64 = (numpy.linalg.norm(
        center_peripheral_axis_difference,
        ord=norm_type,
        axis=axis
    ))



    # 计算正态分布的概率密度函数值
    normal_distribution_value: float = scipy.stats.norm.pdf(
        x=center_peripheral_norm_value,
        loc=mean,
        scale=standard_deviation,
    )
    # 最后一个维度 < 0 反转
    # print("shape:", center_peripheral_axis_difference.shape)
    # 对于每一个其他维度，axis 的第二个值(最后一个维度) < 0 反转
    normal_distribution_value = normal_distribution_value * numpy.where(
        center_peripheral_axis_difference[..., -1] < 0, -1, 1
    )

    return normal_distribution_value


def generate_distribution_kernel(
    kernel_size: int,
    distribution_function: Callable[
        [
            numpy.ndarray,
            numpy.ndarray
        ],
        numpy.ndarray
    ] = functools.partial(
            central_peripheral_weight__norm__normal_distribution,
            norm_type=2,
            mean=0,
            standard_deviation=1,
            axis=-1
        ),
    center_coordinate: numpy.ndarray = numpy.array([0, 0]),
    dtype: numpy.typing.DTypeLike = numpy.float64
) -> numpy.ndarray:
    """
    生成一个符合正态分布的卷积核

    :param kernel_size: 卷积核的大小
    :param distribution_function: 分布函数，默认是正态分布
    :param center_coordinate:
    :param dtype:
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
    coordinate_x_y_grid = numpy.stack(
        [coordinate_x_expand, coordinate_y_expand],
        axis=-1
    )  # (kernel_size, kernel_size, 2)

    # ============================== 计算卷积核 ==============================
    # 计算正态分布的概率密度函数值
    kernel: numpy.ndarray = distribution_function(
            center_coordinate,  # 中心点坐标
            coordinate_x_y_grid  # 周边点坐标
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
    return kernel
