import functools
from typing import Callable, Literal

import numpy
import skimage

from src.lib_common_math.common_math import point_distance_norm
from src.lib_convolution_core.direction_relation import (
    central_peripheral_contrast__opposite_direction_hcl_substraction
)

def central_peripheral_contrast__substraction(
    central_pixel_attributes: numpy.ndarray,
    peripheral_pixel_attributes: numpy.ndarray,
) -> numpy.ndarray:
    """
    计算对比度

    :param central_pixel_attributes: 中心像素的颜色
    :param peripheral_pixel_attributes: 某个周边像素的颜色颜色
    :return:
    """
    colour_contrast: numpy.ndarray = numpy.subtract(
        central_pixel_attributes,
        peripheral_pixel_attributes,
    )

    return colour_contrast


"""
传统的拉普拉斯算子更加接近昆虫的复眼

"""
def central_peripheral_contrast__oklch(
    central_pixel_attributes: numpy.ndarray,
    peripheral_pixel_attributes: numpy.ndarray,
    norm_type: Literal["fro", "nuc"] | int | float | None = None,
    channel_axis: int = -1
) -> numpy.ndarray:
    """
    计算对比度

    :param central_pixel_attributes: 中心像素的颜色
    :param peripheral_pixel_attributes: 某个周边像素的颜色颜色
    :param norm_type: 范数类型，默认为 2 范数
    :param channel_axis: 颜色通道所在的轴
    :return:
    """

    # 计算坐标的范数
    center_peripheral_contrast__oklch: numpy.ndarray = point_distance_norm(
        central_pixel_attributes,
        peripheral_pixel_attributes,
        norm_type=norm_type,
        axis_dimension_index=channel_axis
    )

    # 消除这个维度
    center_peripheral_contrast__oklch = numpy.squeeze(
        center_peripheral_contrast__oklch,
        axis=channel_axis
    )

    return center_peripheral_contrast__oklch


def central_peripheral_contrast__delta_e_cie2000_lch(
    central_pixel_attributes: numpy.ndarray,
    peripheral_pixel_attributes: numpy.ndarray,
    lightness_scale: float = 1,
    chroma_scale: float = 1,
    hue_scale: float = 1,
    channel_axis: int = -1
) -> numpy.ndarray:
    """
    计算对比度

    :param central_pixel_attributes: 中心像素的颜色
    :param peripheral_pixel_attributes: 某个周边像素的颜色颜色
    :param lightness_scale: brightness diff weight 明度差权重
    :param chroma_scale: saturate diff weight 饱和差权重
    :param hue_scale: color type (angle) diff weight 色角差权重
    :param channel_axis: 颜色通道所在的轴
    :return:
    """
    # 转 LCH to Lab
    central_pixel_attributes_lab: numpy.ndarray = skimage.color.lch2lab(
        central_pixel_attributes,
        channel_axis=channel_axis
    )
    peripheral_pixel_attributes_lab: numpy.ndarray = skimage.color.lch2lab(
        peripheral_pixel_attributes,
        channel_axis=channel_axis
    )

    colour_contrast: numpy.ndarray = skimage.color.deltaE_ciede2000(
        lab1=central_pixel_attributes_lab,
        lab2=peripheral_pixel_attributes_lab,
        kL=lightness_scale,
        kC=chroma_scale,
        kH=hue_scale,
        channel_axis=channel_axis,
    )

    return colour_contrast

def central_peripheral_contrast__dynamic_kernel(
    pixel_matrix: numpy.ndarray,
    kernel_size: int,
    cancel_opposite_direction: bool = True,
    central_peripheral_contrast_function: Callable[
        [numpy.ndarray, numpy.ndarray],
        numpy.ndarray
    ] = functools.partial(
        central_peripheral_contrast__oklch,
        norm_type=2,
        channel_axis=-1
    ),
    padding_mode: Literal[
        "constant",
        "edge",
        "linear_ramp",
        "maximum",
        "mean",
        "median",
        "minimum",
        "reflect",
        "symmetric",
        "wrap",
        "empty"
    ] = 'edge',
) -> numpy.ndarray:
    """
    对每个像素点进行卷积，计算其与周边像素的对比度，并返回卷积结果。

    :param pixel_matrix: 输入的像素矩阵，形状为 (height, width, channels)
    :param kernel_size: 卷积核的大小，必须为奇数
    :param cancel_opposite_direction: 是否对向相消，默认True (抛弃渐变 (后面可以加入函数))
    :param central_peripheral_contrast_function: 计算对比度的函数，默认使用 CIEDE2000 颜色差异公式
    :param padding_mode: 填充模式，默认使用 'edge'，即边缘填充
    :return: 卷积结果，形状为 (height, width, channels, kernel_size, kernel_size)
    """
    # ============================= 补边 ==============================
    pad_width = kernel_size // 2
    padded_pixel_matrix: numpy.ndarray = numpy.pad(
        pixel_matrix,
        pad_width=((pad_width, pad_width), (pad_width, pad_width), (0, 0)),
        mode=padding_mode
    )

    # ============================= 卷积展开 ==============================
    unfolded_matrix_combine: numpy.ndarray
    # [height, width, channels] -> [height, width, channels, kernel_size, kernel_size]
    unfolded_matrix_combine = numpy.lib.stride_tricks.sliding_window_view(
        padded_pixel_matrix,
        window_shape=(kernel_size, kernel_size),
        axis=(0, 1)
    )

    if cancel_opposite_direction:
        unfolded_matrix_combine = central_peripheral_contrast__opposite_direction_hcl_substraction(
            unfolded_matrix_combine=unfolded_matrix_combine
        )

    # ============================= 中心像素和周边像素属性 ==============================
    # [height, width, channels, kernel_size, kernel_size]
    central_pixel_attributes: numpy.ndarray
    peripheral_pixel_attributes: numpy.ndarray
    # 每个中心像素的通道和属性
    if kernel_size % 2 == 1:
        central_pixel_attributes = unfolded_matrix_combine[
            :,
            :,
            :,
            pad_width: pad_width + 1,
            pad_width: pad_width + 1,
        ]
    else:
        central_pixel_attributes = numpy.mean(
            unfolded_matrix_combine[
                :,
                :,
                :,
                pad_width - 1 : pad_width + 1,
                pad_width - 1 : pad_width + 1,
            ],
            axis=(-2, -1)
        )

    # 每个周边像素的通道和属性
    peripheral_pixel_attributes = unfolded_matrix_combine

    # 计算对比度之前重排维度，color channel axis to the last axis
    central_pixel_attributes = numpy.moveaxis(
        central_pixel_attributes,
        source=-3,
        destination=-1
    )
    peripheral_pixel_attributes = numpy.moveaxis(
        peripheral_pixel_attributes,
        source=-3,
        destination=-1
    )

    # ============================= 计算对比度 ==============================
    # 计算对比度
    contrast_values: numpy.ndarray = central_peripheral_contrast_function(
        central_pixel_attributes,
        peripheral_pixel_attributes
    )

    # -> [height, width, kernel_size, kernel_size]
    return contrast_values


