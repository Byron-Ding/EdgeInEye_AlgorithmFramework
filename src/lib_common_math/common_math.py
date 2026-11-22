import numpy
from typing import Literal

def point_distance_norm(
    central_pixel_attributes: numpy.ndarray,
    peripheral_pixel_attributes: numpy.ndarray,
    norm_type: Literal["fro", "nuc"] | int | float | None = None,
    axis_dimension_index: int | None = -1
) -> numpy.ndarray:
    """
    计算中心像素属性和周边像素属性 的距离（范数可调）

    :param central_pixel_attributes: 中心像素属性(必须包含坐标)
    :param peripheral_pixel_attributes: 周边像素属性(必须包含坐标)
    :param norm_type: 范数类型，默认为 2 范数
    :param axis_dimension_index: 计算范数的轴，默认为最后一个轴
    :return: 中心像素属性和周边像素属性的差值，以及 范数值
    """

    # ============================== 距离 ==============================
    # 计算坐标的差
    center_peripheral_axis_difference: numpy.ndarray =\
        central_pixel_attributes - peripheral_pixel_attributes

    # print("center shape:", central_pixel_attributes.shape)
    # print("peripheral shape:", peripheral_pixel_attributes.shape)
    # 计算坐标的范数
    center_peripheral_norm_value: numpy.ndarray = numpy.linalg.norm(
        center_peripheral_axis_difference,
        ord=norm_type,
        axis=axis_dimension_index,
        keepdims=True,
    )

    return center_peripheral_norm_value
