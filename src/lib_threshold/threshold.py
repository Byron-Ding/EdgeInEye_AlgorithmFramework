import numpy


def calculate_sigmoid_slope_from_upper_lower_bound(
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
    :param upper_value_x: 上界对应的x值
    :param lower_value_x: 下界对应的x值
    :return: 斜率
    """
    slope = (
        numpy.log(
            upper_bound / (1 - upper_bound)
        )
        - numpy.log(lower_bound / (1 - lower_bound))) / (upper_value_x - lower_value_x)
    return slope


def expand_difference_by_sigmoid_threshold(
        result: numpy.ndarray,
        threshold: float = 1.0,
        slope: float = calculate_sigmoid_slope_from_upper_lower_bound(

        ),
        result_range: tuple[float, float] = (0.0, 255.0)
) -> numpy.ndarray:
    """
    通过sigmoid函数扩展差异。
    这样可以将小的差异放大，使得在阈值附近的变化更加明显，从而增强对比度。

    使用sigmoid函数计算阈值，公式如下：
        f(x) = 1 / (1 + exp(-slope * (x - threshold)))
    其中，slope为斜率，threshold为阈值。
    将sigmoid函数的输出映射到指定的结果范围内。


    :param result: 输入的结果数组
    :param threshold: 阈值
    :param slope: 斜率
    :param result_range: 结果范围
    :return: 感知响应
    """
    sigmoid_result: numpy.ndarray = 1 / (1 + numpy.exp(-slope * (result - threshold)))
    # deltaE: torch.Tensor
    return sigmoid_result * (result_range[1] - result_range[0]) + result_range[0]
