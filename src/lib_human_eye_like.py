# 导入 colour-science 库的 颜色对比度 计算方法
#  对比度 明度 饱和度(彩度)
# 对比度， 色差距离
import functools
from typing import Callable

# from colour.difference import delta_E_CIE2000
import numpy

from src.lib_convolution_core.attention_weight import (
    central_peripheral_weight__norm__normal_distribution,
    generate_distribution_kernel
)
from src.lib_convolution_core.contrast_gradient import (
    central_peripheral_contrast__oklch,
    central_peripheral_contrast__dynamic_kernel
)

"""
构造卷积核，符合拉普拉斯变换。 
总和可以是 0 也可以是 1

需要的参数：
1. kernel_size: 卷积核的大小，必须是奇数
2. initialized_distribution_function: 分布函数，默认是正态分布
    默认：正态分布的注意力


设某个像素点的位置为 (x, y)
则其属性可以表示为
    属性(x, y) = [亮度, 饱和度, 对比度, 色差距离, ...]
则周围各个点的位置
    (x + i, y + j)
        i, j 为横纵方向的偏移量
各个属性为
    属性(x + i, y + j) = [亮度, 饱和度, 对比度, 色差距离, ...]
    注⚠️：可以是列表，也可以是字典，即信息集合
    
传统卷积为
    输出(x, y) = sum_{i, j}{ 属性(x + i, y + j) * kernel(i, j) }
    注⚠️：一般来说，属性(x + i, y + j) 在市面上一般的神经网络中 并不采用向量
        而是只对灰度或者单一的输出值卷积
        即只有单一值的输入
    此处对可以对其进行扩展

本算法模拟人眼，同时扩展计算与中心的偏差：
    人眼会比较余光和中间的注意力，
        但周边亦有权重
        
    于是
        设 中心-周边 比较函数为 对于每个周边 中心周边比较(F(中心)， F(周边))
        
        设 每一对中心-周边 的权重由 坐标(中心) 和 坐标(周边) 或者 属性(中心) 和 属性(周边) 表示
        设其函数为 中心周边权重()
        即 中心-周边 的 综合注意力/权重 值可表示为
        
        属性可以包含坐标，坐标也是一种属性
        
    输出(x, y) = sum_{i, j}{ 
        中心周边权重(**)
        中心周边比较(**) 
    }
"""

central_peripheral_contrast__dynamic_kernel_oklch = functools.partial(
    central_peripheral_contrast__dynamic_kernel,
    cancel_opposite_direction=True,
    central_peripheral_contrast_function=functools.partial(
        central_peripheral_contrast__oklch,
        norm_type=2,
        channel_axis=-1
    ),
    padding_mode='edge'
)

center_peripheral_weight_function_normal_distribution = functools.partial(
    generate_distribution_kernel,
    distribution_function=functools.partial(
        central_peripheral_weight__norm__normal_distribution,
        norm_type=2,
        mean=0,
        standard_deviation=1,
        axis_dimension_index=-1
    ),
    center_coordinate_on_distribution_function=numpy.array([0, 0])
)

def central_peripheral_contrast_convolution_local_normalized(
    pixel_matrix: numpy.ndarray,
    kernel_size: int,
    central_peripheral_contrast_function: Callable[
        [numpy.ndarray, int],
        numpy.ndarray
    ] = central_peripheral_contrast__dynamic_kernel_oklch,
    center_peripheral_weight_function: Callable[
        [int],
        numpy.ndarray
    ] = center_peripheral_weight_function_normal_distribution,
    whether_normalize: bool = False,
    epsilon: float = 1e-8
) -> numpy.ndarray:
    """
    对每个像素点进行卷积，计算其与周边像素的对比度和权重，并返回卷积结果。
        可以加入参数 central_peripheral_weight_function: 中心-周边权重函数

    :param pixel_matrix: 输入的像素矩阵，形状为 (height, width, channels)，其中 channels 表示每个像素的属性维度
    :param kernel_size: 卷积核的大小，可以是奇数或偶数
    :param central_peripheral_contrast_function: 中心-周边对比度函数
    :param center_peripheral_weight_function: 中心-周边权重函数
    :param whether_normalize: 是否对卷积结果进行归一化，默认 False
    :param epsilon: 归一化的时候 防止除零的微小值，默认 1e-8
    :return: 卷积结果矩阵，形状为 (height, width)
    """
    # ============================= 卷积核生成 ==============================
    # 生成卷积核的权重矩阵
    weight_values: numpy.ndarray = center_peripheral_weight_function(
        kernel_size
    )  # shape: (kernel_size, kernel_size)
    # 动态卷积核
    contrast_values_kernel: numpy.ndarray = central_peripheral_contrast_function(
        pixel_matrix,
        kernel_size
    )  # shape: (height, width, kernel_size, kernel_size)

    # ============================= 卷积计算 ==============================
    # 扩展 weight_values 的维度以便与 contrast_values_kernel 相乘
    weight_values_expanded: numpy.ndarray = numpy.expand_dims(
        weight_values, axis=(0, 1)
    )  # shape: (1, 1, kernel_size, kernel_size)

    # 将 contrast_values_kernel 和 weight_values_expanded 相乘 （每一位的权重乘以对应的对比度值）
    # shape: [height, width, kernel_size, kernel_size]
    contrast_values_kernel_weighted: numpy.ndarray = contrast_values_kernel * weight_values_expanded

    # ============================= 卷积结果计算 ==============================
    # 将 contrast_values_kernel_weighted 在最后两个维度上求和，得到卷积结果
    contrast_dynamic_convolution_result: numpy.ndarray = numpy.sum(
        contrast_values_kernel_weighted, axis=(-1, -2)
    )  # shape: [height, width]

    # ============================= 归一化 ==============================
    if whether_normalize:
        # 归一化卷积结果到 [0, 1] 范围
        contrast_dynamic_convolution_result_min: float = numpy.min(
            contrast_dynamic_convolution_result
        )
        contrast_dynamic_convolution_result_max: float = numpy.max(
            contrast_dynamic_convolution_result
        )
        contrast_dynamic_convolution_result_range: float = (
            contrast_dynamic_convolution_result_max
            - contrast_dynamic_convolution_result_min
            + epsilon
        )
        contrast_dynamic_convolution_result = (
                    contrast_dynamic_convolution_result
                - contrast_dynamic_convolution_result_min
        ) / contrast_dynamic_convolution_result_range

    return contrast_dynamic_convolution_result


