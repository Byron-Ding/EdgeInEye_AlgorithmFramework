import numpy

"""
算法:
如果是边缘补齐
数值同化算法:
    ori: a < b
    a 同化 b (拉低),
    b_{new} = ln(1 + exp(b - a)) + a
直线天然有延长的倾向
差距天然会往小的方向走
直线边缘内部差异非常小，有一边大了说明断了，需要拉低。
（可迭代）


但是目前使用感受野算法

"""



"""
另外就是可以对向相消 也可以不对向相消
对向相消
"""
def central_peripheral_contrast__opposite_direction_hcl_substraction(
    unfolded_matrix_combine: numpy.ndarray,    # [height, width, channels, kernel_size, kernel_size]
) -> numpy.ndarray:
    """
    防止连续的对比度变化 也就是渐变被判断为边缘，这里可以进行阈值计算，拉伸减法，后续扩展

    :param unfolded_matrix_combine: unfloded_matrix_combine, shape: [height, width, channels, kernel_size, kernel_size]
    :return: unfolded_matrix_combine after opposite direction HCL subtraction
    """
    # 对向相消
    # -3 color channel, -2 kernel height, -1 kernel width
    # L,   C,H
    out__unfolded_matrix_combine: numpy.ndarray = unfolded_matrix_combine.copy()

    out__flipped_unfolded_matrix_combine: numpy.ndarray = numpy.flip(out__unfolded_matrix_combine, axis=(-1, -2))
    out__unfolded_matrix_combine[..., 0, :, :] = numpy.abs(
        out__unfolded_matrix_combine[..., 0, :, :] - out__flipped_unfolded_matrix_combine[..., 0, :, :]
    )

    # Calculate CH (r, theta) difference
    r1: numpy.ndarray = out__unfolded_matrix_combine[..., 1, :, :]
    theta1: numpy.ndarray = out__unfolded_matrix_combine[..., 2, :, :]
    r2: numpy.ndarray = out__flipped_unfolded_matrix_combine[..., 1, :, :]
    theta2: numpy.ndarray = out__flipped_unfolded_matrix_combine[..., 2, :, :]

    # Change to Complex representation for polar coordinates
    complex1: numpy.ndarray = r1 * numpy.exp(1j * theta1)
    complex2: numpy.ndarray = r2 * numpy.exp(1j * theta2)

    # Calculate the difference in complex space
    complex_difference: numpy.ndarray = complex1 - complex2

    # Convert back to polar coordinates
    r_difference: numpy.ndarray = numpy.abs(complex_difference)
    theta_difference: numpy.ndarray = numpy.angle(complex_difference)

    # Update the unfolded matrix with the differences
    out__unfolded_matrix_combine[..., 1, :, :] = r_difference
    out__unfolded_matrix_combine[..., 2, :, :] = theta_difference

    return out__unfolded_matrix_combine