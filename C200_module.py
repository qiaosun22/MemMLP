# =============================== #
# @Author  : Wang Ze
# @Time    : 2022/5/18
# @Software: PyCharm
# @File    : C200_module.py
# =============================== #
from C200_utils import *
import matplotlib.pyplot as plt

# ======================= #
# 函数形式计算层
# ======================= #
# 池化 fast
def pooling(feature_map, kernel_size):
    channels = feature_map.shape[0]
    pooled_rows = int(feature_map.shape[1] / kernel_size)
    pooled_cols = int(feature_map.shape[2] / kernel_size)
    output = feature_map.reshape(channels, pooled_rows, kernel_size, pooled_cols, kernel_size)
    output = output.max(axis = (2, 4))
    return output


# 144k 片上推理卷积封装 函数形式
def conv2d_144k(sdk, input_feature_map, weight_addr, repeat,
                stride, kernel_size, padding,
                input_half_level, output_half_level,
                it_time = 10,
                relu = True,
                input_quant = False):
    # ================================= #
    # 参数说明
    # ================================= #
    # input_feature_map:
    #   输入 feature map, 矩阵大小为 [C, H, W]
    # weight_addr:
    #   144K 上复制后的权重, 矩阵大小为 [rows * repeat[0], cols * repeat[1]]
    # repeat:
    #   144K 上复制权重的次数, repeat[0] 为行复制次数, repeat[1] 为列复制次数
    # stride, kernel_size, padding：
    #   卷积相关参数
    # input_half_level:
    #   输入数据量化等级
    # output_half_level:
    #   输出数据量化等级
    # relu:
    #   是否 relu
    # input_quant:
    #   是否对输入数据进行量化

    # 补齐维度
    while len(input_feature_map.shape) < 3:
        input_feature_map = np.expand_dims(input_feature_map, axis = 0)
    # 计算输出大小
    _, input_rows, input_cols = input_feature_map.shape
    out_feature_size_rows = int((input_rows + 2 * padding - kernel_size) / stride + 1)
    out_feature_size_cols = int((input_cols + 2 * padding - kernel_size) / stride + 1)

    # 输入数据量化
    if input_quant:
        input_feature_map, _ = data_quantization_sym(input_feature_map, half_level = input_half_level,
                                                     isint = 1)
    # 输入图像重排
    array_input = feature_map_to_input(input_feature_map, stride = stride, kernel_size = kernel_size,
                                       padding = padding, repeat = repeat)
    # 乘加运算(卷积 f1), 使用脉冲展开
    array_output = mvm_bitwise_concat_push_fast_144k(sdk, array_input, weight_addr, repeat, it_time = it_time)
    # Relu
    if relu:
        array_output[array_output < 0] = 0
    # 数据量化
    array_output, _ = data_quantization_sym(array_output, half_level = output_half_level, isint = 1)
    # 数据重排
    array_output = output_to_feature_map(array_output, out_feature_size_rows, out_feature_size_cols)

    return array_output


# 144k 片上推理全连接封装 函数形式
def linear_144k(sdk, input_feature_map, weight_addr, repeat,
                input_half_level, output_half_level,
                it_time = 10,
                relu = True,
                input_quant = False):
    # ================================= #
    # 参数说明
    # ================================= #
    # input_feature_map:
    #   输入 feature map, 矩阵大小为 [C, H, W]
    # weight_addr:
    #   144K 上复制后的权重, 矩阵大小为 [rows * repeat[0], cols * repeat[1]]
    # repeat:
    #   144K 上复制权重的次数, repeat[0] 为行复制次数, repeat[1] 为列复制次数
    # input_half_level:
    #   输入数据量化等级
    # output_half_level:
    #   输出数据量化等级
    # relu:
    #   是否 relu
    # input_quant:
    #   是否对输入数据进行量化
    array_input = input_feature_map.reshape(-1, 1)
    if input_quant:
        array_input, _ = data_quantization_sym(array_input, half_level = input_half_level, isint = 1)
    array_input = np.tile(array_input, [repeat[0], 1])

    array_output = mvm_bitwise_concat_push_fast_144k(sdk, array_input, weight_addr, repeat, it_time = it_time)
    if relu:
        array_output[array_output < 0] = 0
    array_output, _ = data_quantization_sym(array_output, half_level = output_half_level, isint = 1)

    return array_output



# 144k 片上推理卷积封装 函数形式
def conv2d_sim(sdk, input_feature_map, weights, repeat,
                stride, kernel_size, padding,
                input_half_level, output_half_level,
                it_time = 10,
                relu = True,
                input_quant = False):
    # ================================= #
    # 参数说明
    # ================================= #
    # input_feature_map:
    #   输入 feature map, 矩阵大小为 [C, H, W]
    # weight_addr:
    #   144K 上复制后的权重, 矩阵大小为 [rows * repeat[0], cols * repeat[1]]
    # repeat:
    #   144K 上复制权重的次数, repeat[0] 为行复制次数, repeat[1] 为列复制次数
    # stride, kernel_size, padding：
    #   卷积相关参数
    # input_half_level:
    #   输入数据量化等级
    # output_half_level:
    #   输出数据量化等级
    # relu:
    #   是否 relu
    # input_quant:
    #   是否对输入数据进行量化

    # 补齐维度
    while len(input_feature_map.shape) < 3:
        input_feature_map = np.expand_dims(input_feature_map, axis = 0)
    # 计算输出大小
    _, input_rows, input_cols = input_feature_map.shape
    out_feature_size_rows = int((input_rows + 2 * padding - kernel_size) / stride + 1)
    out_feature_size_cols = int((input_cols + 2 * padding - kernel_size) / stride + 1)

    # 输入数据量化
    if input_quant:
        input_feature_map, _ = data_quantization_sym(input_feature_map, half_level = input_half_level,
                                                     isint = 1)
    # 输入图像重排
    array_input = feature_map_to_input(input_feature_map, stride = stride, kernel_size = kernel_size,
                                       padding = padding, repeat = repeat)
    # 模拟乘加运算
    array_output = mvm_bitwise_concat_push_fast(array_input, weights, repeat)
    # Relu
    if relu:
        array_output[array_output < 0] = 0
    # 数据量化
    array_output, _ = data_quantization_sym(array_output, half_level = output_half_level, isint = 1)
    # 数据重排
    array_output = output_to_feature_map(array_output, out_feature_size_rows, out_feature_size_cols)

    return array_output


# 144k 片上推理全连接封装 函数形式
def linear_sim(sdk, input_feature_map, weights, repeat,
                input_half_level, output_half_level,
                it_time = 10,
                relu = True,
                input_quant = False):
    # ================================= #
    # 参数说明
    # ================================= #
    # input_feature_map:
    #   输入 feature map, 矩阵大小为 [C, H, W]
    # weight_addr:
    #   144K 上复制后的权重, 矩阵大小为 [rows * repeat[0], cols * repeat[1]]
    # repeat:
    #   144K 上复制权重的次数, repeat[0] 为行复制次数, repeat[1] 为列复制次数
    # input_half_level:
    #   输入数据量化等级
    # output_half_level:
    #   输出数据量化等级
    # relu:
    #   是否 relu
    # input_quant:
    #   是否对输入数据进行量化
    array_input = input_feature_map.reshape(-1, 1)
    if input_quant:
        array_input, _ = data_quantization_sym(array_input, half_level = input_half_level, isint = 1)
    array_input = np.tile(array_input, [repeat[0], 1])

    array_output = mvm_bitwise_concat_push_fast(array_input, weights, repeat)
    if relu:
        array_output[array_output < 0] = 0
    array_output, _ = data_quantization_sym(array_output, half_level = output_half_level, isint = 1)

    return array_output