# import functools

# 显示卷积结果
import matplotlib.pyplot
import numpy
# save
import skimage
from skimage import color, io


# from src.lib_convolution_core.attention_weight import (
#     central_peripheral_weight__norm__normal_distribution,
#     generate_distribution_kernel
# )
#
# from src.lib_convolution_core.contrast_gradient import (
#     central_peripheral_contrast__delta_e_cie2000_lch,
#     central_peripheral_contrast__oklch,
#     central_peripheral_contrast__dynamic_kernel
# )


from src.lib_threshold.threshold import (
    calculate_sigmoid_slope_from_upper_lower_bound,
    expand_difference_by_sigmoid_threshold
)

from src.lib_human_eye_like import (
    central_peripheral_contrast_convolution_local_normalized
)



def main() -> None:

    image = io.imread("./img/img.png")  # 读取图片
    image_lab = color.rgb2lab(image)  # 转换为 Lab 颜色空间

    # # 计算颜色平均值
    # image_lab_mean = numpy.mean(image_lab, axis=(0, 1))
    # # 归一化颜色平均值
    # image_lab_mean = image_lab_mean / 255.0  # 将平均值归一化到 [0, 1] 范围

    kernel_size = 5

    # 计算卷积结果
    convolution_result = central_peripheral_contrast_convolution_local_normalized(
        pixel_matrix=image_lab,
        kernel_size=kernel_size,
    )

    matplotlib.pyplot.imshow(convolution_result, cmap='gray')
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.show()

    # 转为 uint8 类型
    convolution_result_save: numpy.ndarray
    convolution_result_save = convolution_result.astype(numpy.uint8)
    io.imsave("./img/convolution_result_initial.png", convolution_result_save)  # 保存卷积结果

    slope: float = calculate_sigmoid_slope_from_upper_lower_bound(
        upper_bound=0.95,
        lower_bound=0.05,
        upper_value_x=20,
        lower_value_x=10
    )

    # 阈值
    convolution_result = expand_difference_by_sigmoid_threshold(
        result=convolution_result,
        threshold=12,
        slope=slope,  # --math.log(0.1 / 1.9),
        result_range=(0, 255)
    )

    # 显示卷积结果
    matplotlib.pyplot.imshow(convolution_result, cmap='gray')
    matplotlib.pyplot.axis('off')
    matplotlib.pyplot.show()

    # 转为 uint8 类型
    convolution_result_save = convolution_result.astype(numpy.uint8)
    skimage.io.imsave("./img/convolution_result_thresholded.png", convolution_result_save)  # 保存卷积结果


if __name__ == "__main__":
    main()
