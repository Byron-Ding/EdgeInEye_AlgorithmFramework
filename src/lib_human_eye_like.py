# 导入 colour-science 库的 颜色对比度 计算方法
#  对比度 明度 饱和度(彩度)
# 对比度， 色差距离
from typing import Callable, cast, Literal

# from colour.difference import delta_E_CIE2000
import numpy
import scipy
import skimage.color
import torch
import functools
import math

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

def general_central_peripheral_weight(
    central_pixel_attributes: numpy.ndarray | torch.Tensor,
    peripheral_pixel_attributes: numpy.ndarray | torch.Tensor,
    central_peripheral_weight_function: Callable[[numpy.ndarray | torch.Tensor, numpy.ndarray | torch.Tensor], float]
) -> float:
    """
    约定：
    1. 中心像素属性为 central_pixel_attributes
    2. 周边像素属性为 peripheral_pixel_attributes
    3. 中心像素属性和周边像素属性的维度相同
    4. 以上属性的前两位为坐标，后续为其他属性
    """

    return central_peripheral_weight_function(central_pixel_attributes, peripheral_pixel_attributes)



def general_central_peripheral_contrast(
    central_pixel_attributes: numpy.ndarray | torch.Tensor,
    peripheral_pixel_attributes: numpy.ndarray | torch.Tensor,
    central_peripheral_contrast_function: Callable[[numpy.ndarray | torch.Tensor, numpy.ndarray | torch.Tensor], float]
) -> float | numpy.ndarray | torch.Tensor:
    """
    约定：
    1. 中心像素属性为 central_pixel_attributes
    2. 周边像素属性为 peripheral_pixel_attributes
    3. 中心像素属性和周边像素属性的维度相同
    4. 以上属性的前两位为坐标，后续为其他属性
    """

    return central_peripheral_contrast_function(central_pixel_attributes, peripheral_pixel_attributes)

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

    # 生成坐标矩阵
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

    # meshgrid 意思是沿着 各自的 方向扩展
    coordinate_x: numpy.ndarray
    coordinate_y: numpy.ndarray
    coordinate_x_expand, coordinate_y_expand = numpy.meshgrid(coordinates, coordinates, indexing="ij")

    # stack 在第N个维度上堆叠矩阵
    grid = numpy.stack([coordinate_x_expand, coordinate_y_expand], axis=-1)  # (kernel_size, kernel_size, 2)


    # 计算正态分布的概率密度函数值
    kernel: numpy.ndarray | torch.Tensor  = cast(
        numpy.ndarray | torch.Tensor,
        distribution_function(
            center_coordinate,  # 中心点坐标
            grid  # 周边点坐标
        )
    )

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


def central_peripheral_contrast__delta_e_cie2000(
    central_pixel_attributes: numpy.ndarray | torch.Tensor,
    peripheral_pixel_attributes: numpy.ndarray | torch.Tensor,
    lightness_scale: float = 1,
    chroma_scale: float = 1,
    hue_scale: float = 1,
    channel_axis: int = -1
) -> torch.Tensor | float | numpy.ndarray:
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
    colour_contrast: torch.Tensor | float | numpy.ndarray = skimage.color.deltaE_ciede2000(
        lab1=central_pixel_attributes,
        lab2=peripheral_pixel_attributes,
        kL=lightness_scale,
        kC=chroma_scale,
        kH=hue_scale,
        channel_axis=channel_axis,
    )

    return colour_contrast

# TODO: 动态生成 卷积核，符合拉普拉斯变换。 支持偶数
# TODO: 使用 torch unfold然后卷积。
def central_peripheral_contrast_convolution(
    pixel_matrix: numpy.ndarray | torch.Tensor,
    kernel_size: int,
    central_peripheral_contrast_function: Callable[[numpy.ndarray | torch.Tensor, numpy.ndarray | torch.Tensor], float | numpy.ndarray | torch.Tensor] =\
        functools.partial(
            central_peripheral_contrast__delta_e_cie2000,
            lightness_scale=1,
            chroma_scale=1,
            hue_scale=1
        ),
    center_peripheral_weight_function: Callable[[...],
        torch.Tensor | float | numpy.ndarray
    ] = functools.partial(
        generate_distribution_kernel,
        kernel_size=3,
        distribution_function=functools.partial(
            central_peripheral_weight__norm__normal_distribution,
            norm_type=2,
            mean=0,
            standard_deviation=1,
            axis=-1
        ),
        center_coordinate=numpy.array([0, 0])
    )

) -> numpy.ndarray:
    """
    对每个像素点进行卷积，计算其与周边像素的对比度和权重，并返回卷积结果。
        可以加入参数 central_peripheral_weight_function: 中心-周边权重函数

    :param pixel_matrix: 输入的像素矩阵，形状为 (height, width, channels)，其中 channels 表示每个像素的属性维度
    :param kernel_size: 卷积核的大小，可以是奇数或偶数
    :param central_peripheral_contrast_function: 中心-周边对比度函数
    :param center_peripheral_weight_function: 中心-周边权重函数
    :return: 卷积结果矩阵，形状为 (height, width)
    """
    # 展开矩阵
    unfolded_matrix_combine: torch.Tensor = torch.nn.functional.unfold(
        torch.tensor(pixel_matrix).permute(2, 0, 1).unsqueeze(0),  # (1, channels, height, width)
        kernel_size=kernel_size,
        padding=kernel_size // 2
    )  # (1, channels * kernel_size * kernel_size, height * width)

    unfolded_matrix: torch.Tensor = unfolded_matrix_combine.view(
        1,
        pixel_matrix.shape[2],  # channels
        kernel_size,
        kernel_size,
        pixel_matrix.shape[0],  # height
        pixel_matrix.shape[1]   # width
    )  # (1, channels, kernel_size, kernel_size, height, width)


    # 广播运算
    # 每个中心像素的通道和属性
    # (1, channels, height, width)
    central_pixel_attributes: torch.Tensor = unfolded_matrix[:, :, kernel_size // 2, kernel_size // 2, :, :]
    # 每个周边像素的通道和属性
    # (1, channels, kernel_size, kernel_size, height, width)
    peripheral_pixel_attributes: torch.Tensor = unfolded_matrix

    # (1, height, width, channels, 1, 1)
    central_pixel_attributes_expanded: torch.Tensor = central_pixel_attributes.permute(0, 2, 3, 1).unsqueeze(4).unsqueeze(5)
    # (1, height, width, channels, kernel_size, kernel_size)
    peripheral_pixel_attributes_expanded: torch.Tensor = peripheral_pixel_attributes.permute(0, 4, 5, 1, 2, 3)

    print(central_pixel_attributes_expanded.shape)
    print(peripheral_pixel_attributes_expanded.shape)

    # 计算对比度和权重
    # (1, height, width, kernel_size, kernel_size)
    contrast_values: torch.Tensor = torch.Tensor(central_peripheral_contrast_function(
        central_pixel_attributes_expanded,
        peripheral_pixel_attributes_expanded
    ))
    weight_values: numpy.ndarray = cast(
        numpy.ndarray,
        center_peripheral_weight_function()
    )

    # contrast_values = perceptual_response(
    #     contrast_values,
    #     threshold=2,
    #     slope=1.5
    # )

    print(contrast_values.shape)
    print(weight_values.shape)


    # 计算卷积结果
    convolution_result: torch.Tensor = torch.sum(
        contrast_values * torch.tensor(weight_values).unsqueeze(0).unsqueeze(0),
        dim=(3, 4)
    )  # (1, height, width)

    return convolution_result.squeeze(0).numpy()  # (height, width)

def central_peripheral_contrast_convolution_local_normalized(
    pixel_matrix: numpy.ndarray | torch.Tensor,
    kernel_size: int,
    central_peripheral_contrast_function: Callable[[numpy.ndarray | torch.Tensor, numpy.ndarray | torch.Tensor], float | numpy.ndarray | torch.Tensor] =\
        functools.partial(
            central_peripheral_contrast__delta_e_cie2000,
            lightness_scale=1,
            chroma_scale=1,
            hue_scale=1
        ),
    center_peripheral_weight_function: Callable[[...],
        torch.Tensor | float | numpy.ndarray
    ] = functools.partial(
        generate_distribution_kernel,
        kernel_size=3,
        distribution_function=functools.partial(
            central_peripheral_weight__norm__normal_distribution,
            norm_type=2,
            mean=0,
            standard_deviation=1,
            axis=-1
        ),
        center_coordinate=numpy.array([0, 0])
    )
) -> numpy.ndarray:
    """
    对每个像素点进行卷积，计算其与周边像素的对比度和权重，并返回卷积结果。
        可以加入参数 central_peripheral_weight_function: 中心-周边权重函数

    :param pixel_matrix: 输入的像素矩阵，形状为 (height, width, channels)，其中 channels 表示每个像素的属性维度
    :param kernel_size: 卷积核的大小，可以是奇数或偶数
    :param central_peripheral_contrast_function: 中心-周边对比度函数
    :param center_peripheral_weight_function: 中心-周边权重函数
    :return: 卷积结果矩阵，形状为 (height, width)
    """
    # 展开矩阵
    unfolded_matrix_combine: torch.Tensor = torch.nn.functional.unfold(
        torch.tensor(pixel_matrix).permute(2, 0, 1).unsqueeze(0),  # (1, channels, height, width)
        kernel_size=kernel_size,
        padding=kernel_size // 2
    )  # (1, channels * kernel_size * kernel_size, height * width)

    unfolded_matrix: torch.Tensor = unfolded_matrix_combine.view(
        1,
        pixel_matrix.shape[2],  # channels
        kernel_size,
        kernel_size,
        pixel_matrix.shape[0],  # height
        pixel_matrix.shape[1]   # width
    )  # (1, channels, kernel_size, kernel_size, height, width)


    # 广播运算
    # 每个中心像素的通道和属性
    # (1, channels, height, width)
    central_pixel_attributes: torch.Tensor = unfolded_matrix[:, :, kernel_size // 2, kernel_size // 2, :, :]
    # 每个周边像素的通道和属性
    # (1, channels, kernel_size, kernel_size, height, width)
    peripheral_pixel_attributes: torch.Tensor = unfolded_matrix

    # (1, height, width, channels, 1, 1)
    central_pixel_attributes_expanded: torch.Tensor = central_pixel_attributes.permute(0, 2, 3, 1).unsqueeze(4).unsqueeze(5)
    # (1, height, width, channels, kernel_size, kernel_size)
    peripheral_pixel_attributes_expanded: torch.Tensor = peripheral_pixel_attributes.permute(0, 4, 5, 1, 2, 3)

    print(central_pixel_attributes_expanded.shape)
    print(peripheral_pixel_attributes_expanded.shape)


    # 计算对比度和权重
    # (1, height, width, kernel_size, kernel_size)
    contrast_values: torch.Tensor = torch.Tensor(central_peripheral_contrast_function(
        central_pixel_attributes_expanded,
        peripheral_pixel_attributes_expanded
    ))


    weight_values: numpy.ndarray = cast(
        numpy.ndarray,
        center_peripheral_weight_function()
    )

    print(contrast_values.shape)
    print(weight_values.shape)


    # 根据当前窗口内部的 contrast 进行归一化
    # 计算当前窗口内部的最大值和最小值
    contrast_values_kernel_flattened: torch.Tensor = contrast_values.view(
        contrast_values.shape[0],  # batch size
        contrast_values.shape[1],  # height
        contrast_values.shape[2],  # width
        contrast_values.shape[3],  # channels
        -1  # flatten kernel_size * kernel_size
    )  # (1, height, width, channels, kernel_size * kernel_size)
    # max_values: torch.Tensor = torch.max(contrast_values_kernel_flattened, dim=4, keepdim=True).values
    # min_values: torch.Tensor = torch.min(contrast_values_kernel_flattened, dim=4, keepdim=True).values
    # print(max_values.shape)
    # print(min_values.shape)
    # print(contrast_values_kernel_flattened.shape)
    # contrast_values = (contrast_values - min_values) / (max_values - min_values + 1e-8)  # 避免除以零

    # 计算卷积结果
    convolution_result: torch.Tensor = torch.sum(
        contrast_values_kernel_flattened * torch.tensor(weight_values).unsqueeze(0).unsqueeze(0),
        dim=(3, 4)
    )  # (1, height, width)

    return convolution_result.squeeze(0).numpy()  # (height, width)


def central_peripheral_contrast__substraction(
    central_pixel_attributes: numpy.ndarray | torch.Tensor,
    peripheral_pixel_attributes: numpy.ndarray | torch.Tensor,
) -> torch.Tensor | float | numpy.ndarray:
    """
    计算对比度

    :param central_pixel_attributes: 中心像素的颜色
    :param peripheral_pixel_attributes: 某个周边像素的颜色颜色
    :param channel_axis: 颜色通道所在的轴
    :return:
    """
    colour_contrast: torch.Tensor | float | numpy.ndarray = central_pixel_attributes - peripheral_pixel_attributes

    return colour_contrast


def perceptual_response(
    result: float | torch.Tensor | numpy.ndarray,
    threshold: float = 2.0,
    slope: float = math.log(0.1 / 1.9)
):
    sigmoid_result = 1 / (1 + torch.exp(-slope * (result - threshold)))
    # deltaE: torch.Tensor
    return sigmoid_result * 255


def calculate_slope_from_upper_lower_bound(
    upper_bound: float= 0.95,
    lower_bound: float= 0.05,
    upper_value_x: float= 1.5,
    lower_value_x: float= 0.5
) -> float:
    """
    根据上界和下界计算斜率
    sigmoid函数的公式为：
        f(x) = 1 / (1 + exp(-slope * (x - threshold)))
    其中，slope为斜率，threshold为阈值。
    当x=upper_value_x时，f(x)=upper_bound
    当x=lower_value_x时，f(x)=lower_bound

    :param upper_bound: 上界
    :param lower_bound: 下界
    :return: 斜率
    """
    slope = (math.log(upper_bound / (1 - upper_bound)) - math.log(lower_bound / (1 - lower_bound))) / (upper_value_x - lower_value_x)
    return slope



if __name__ == '__main__':
    import math
    #
    #
    # central_peripheral_contrast__delta_e_cie2000_value = central_peripheral_contrast__delta_e_cie2000(
    #     central_pixel_attributes=numpy.array([50.0, 50.0, 50.0]),
    #     peripheral_pixel_attributes=numpy.array([60.0, 60.0, 60.0]),
    #     lightness_scale=1,
    #     chroma_scale=1,
    #     hue_scale=1
    # )
    # print(central_peripheral_contrast__delta_e_cie2000_value)
    #
    #
    # central_peripheral_weight__norm__normal_distribution_value = central_peripheral_weight__norm__normal_distribution(
    #     central_pixel_attributes=numpy.array([10, 10]),
    #     peripheral_pixel_attributes=numpy.array([11, 11]),
    #     norm_type=2,
    #     mean=0,
    #     standard_deviation=1
    # )
    # print(central_peripheral_weight__norm__normal_distribution_value)
    #
    # # norm(根号二）
    # print(
    #     scipy.stats.norm.pdf(
    #         x=math.sqrt(2),
    #         loc=0,
    #         scale=1
    #     )
    # )

    # print(generate_distribution_kernel(kernel_size=3))

    # 读图片
    from skimage import io, color
    image = io.imread("../img/img.png")  # 读取图片
    image_lab = color.rgb2lab(image)  # 转换为 Lab 颜色空间

    # 计算颜色平均值
    image_lab_mean = numpy.mean(image_lab, axis=(0, 1))
    # 归一化颜色平均值
    image_lab_mean = image_lab_mean / 255.0  # 将平均值归一化到 [0, 1] 范围

    # 计算卷积结果
    convolution_result = central_peripheral_contrast_convolution(
        pixel_matrix=image_lab,
        kernel_size=7,
        central_peripheral_contrast_function=functools.partial(
            central_peripheral_contrast__delta_e_cie2000,
            lightness_scale=1,
            chroma_scale=1,
            hue_scale=1,
            channel_axis=3
        ),
        center_peripheral_weight_function=functools.partial(
            generate_distribution_kernel,
            kernel_size=7,
            distribution_function=functools.partial(
                central_peripheral_weight__norm__normal_distribution,
                norm_type=math.inf,
                mean=0,
                standard_deviation=0.5,
                axis=-1
            ),
            center_coordinate=numpy.array([0, 0])
        )
    )
    
    # 显示卷积结果

    # 显示卷积结果
    import matplotlib.pyplot as plt
    plt.imshow(convolution_result, cmap='gray')
    plt.axis('off')
    plt.show()
    # save
    import skimage.io
    # 转为 uint8 类型
    convolution_result = convolution_result.astype(numpy.uint8)
    skimage.io.imsave("../img/convolution_result_original.png", convolution_result)  # 保存卷积结果

    convolution_result_small = perceptual_response(
        torch.tensor(convolution_result),
        threshold=1.5,
        slope=1000 # --math.log(0.1 / 1.9)
    )


    convolution_result = convolution_result - perceptual_response(
        torch.tensor(convolution_result_small),
        threshold=1.0,
        slope=1000 # --math.log(0.1 / 1.9)
    )

    # to 0 - 255
    # convolution_result = (convolution_result - torch.min(convolution_result)) / (torch.max(convolution_result) - torch.min(convolution_result)) * 255
    convolution_result = convolution_result.numpy()

    # 显示卷积结果
    import matplotlib.pyplot as plt
    plt.imshow(convolution_result, cmap='gray')
    plt.axis('off')
    plt.show()
    # save
    import skimage.io
    # 转为 uint8 类型
    convolution_result = convolution_result.astype(numpy.uint8)
    skimage.io.imsave("../img/convolution_result.png", convolution_result)  # 保存卷积结果



