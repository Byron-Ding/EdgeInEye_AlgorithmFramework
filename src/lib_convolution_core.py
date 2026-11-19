from typing import Callable, cast, Literal
import functools
import scipy.stats

import numpy
import torch



def central_peripheral_weight__norm__normal_distribution(
    central_pixel_attributes: numpy.ndarray | torch.Tensor,
    peripheral_pixel_attributes: numpy.ndarray | torch.Tensor,
    norm_type: Literal["fro", "nuc"] | int | float | None = None,
    mean: float = 0.0,
    standard_deviation: float = 1.0,
    axis: int | None = -1
) -> float:
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
    distribution_function: Callable[[numpy.ndarray | torch.Tensor, numpy.ndarray | torch.Tensor], float | numpy.ndarray | torch.Tensor] =\
        functools.partial(
            central_peripheral_weight__norm__normal_distribution,
            norm_type=2,
            mean=0,
            standard_deviation=1,
            axis=-1
        ),
    center_coordinate: numpy.ndarray | torch.Tensor = numpy.array([0, 0])
) -> numpy.ndarray:
    """
    生成一个符合正态分布的卷积核

    :param kernel_size: 卷积核的大小
    :param distribution_function: 分布函数，默认是正态分布
    :return: 卷积核矩阵，形状为 (kernel_size, kernel_size)
    """

    # =============================== 生成一维坐标 ===============================
    if kernel_size % 2 == 1:  # 奇数
        coordinates: numpy.ndarray = numpy.arange(
            -(kernel_size // 2),
            stop=kernel_size // 2 + 1
        )
    else:  # 偶数
        coordinates = numpy.arange(
            -(kernel_size / 2) + 0.5,
            stop=kernel_size / 2,
            step=1.0
        )

    # =============================== 生成 ===============================
    # meshgrid 意思是沿着 各自的 方向扩展
    # 比如
    # coordinates = [1, 2, 3]
    # meshgrid(coordinates, coordinates, indexing="ij") 会生成
    # coordinate_x = [[1, 1, 1],
    #                 [2, 2, 2],
    #                 [3, 3, 3]]
    # coordinate_y = [[1, 2, 3],
    #                 [1, 2, 3],
    #                 [1, 2, 3]]
    coordinate_x: numpy.ndarray
    coordinate_y: numpy.ndarray
    coordinate_x_expand, coordinate_y_expand = numpy.meshgrid(coordinates, coordinates, indexing="ij")

    # 在最后一个维度上堆叠坐标矩阵，形成一个形状为 (kernel_size, kernel_size, 2) 的网格坐标矩阵
    # stack 在第N个维度上堆叠矩阵
    grid = numpy.stack([coordinate_x_expand, coordinate_y_expand], axis=-1)  # (kernel_size, kernel_size, 2)

    # ============================== 计算卷积核 ==============================
    # 计算正态分布的概率密度函数值
    kernel: numpy.ndarray | torch.Tensor  = cast(
        numpy.ndarray | torch.Tensor,
        distribution_function(
            center_coordinate,  # 中心点坐标
            grid  # 周边点坐标
        )
    )

    # ============================== 调整中心值 ==============================
    # 中心是一个大负数，为所有东西（中心除外）的和
    original_center_value: float = cast(
        float,
        distribution_function(
            center_coordinate,  # 中心点坐标
            center_coordinate  # 中心点坐标
        )
    )
    if kernel_size % 2 == 1:
        kernel[kernel_size // 2, kernel_size // 2] = -numpy.sum(kernel) + original_center_value + 1
    else:
        # 改变中心四个方块
        kernel[kernel_size // 2 - 1:kernel_size // 2 + 1, kernel_size // 2 - 1:kernel_size // 2 + 1] +=\
            (-numpy.sum(kernel) + original_center_value + 1) / 4

    return kernel
