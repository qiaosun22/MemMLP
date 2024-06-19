# from sim_utils import *
import time
import numpy as np


def scale_to_ascii(value):
    # 定义一个字符列表，字符的位置对应于像素值的大小
    ASCII_CHARS = '@%#*+=-:. '
    """把像素值转换为ascii字符"""
    return ASCII_CHARS[len(ASCII_CHARS) - 1 - int(value * (len(ASCII_CHARS) - 1))]


def plot_ascii_img(img, spacing = 0, boarder = True):
    row, col = img.shape
    blank = ' ' * spacing
    boarder_str = f'-' * (col * (spacing + 1) + 2)
    if boarder:
        print(boarder_str)
    for row in img:
        row_str = ''.join(f'{scale_to_ascii(val)}{blank}' for val in row)
        if boarder:
            row_str = f'|{row_str}|'
        print(row_str)
    if boarder:
        print(boarder_str)


# ======================================== #
# Functions related to 144K simulation
# ======================================== #
# data quant
def data_quantization_sym(data_float, half_level = 15, scale = None, isint = 0, clamp_std = None):
    # isint = 1 -> return quantized values as integer levels
    # isint = 0 -> return quantized values as float numbers with the same range as input
    if half_level <= 0:
        return data_float, 0

    std = data_float.std()
    # if clamp_std != None and clamp_std != 0:
    #     data_float = torch.clamp(data_float, min = -clamp_std * std, max = clamp_std * std)

    if scale == None or scale == 0:
        scale = abs(data_float).max()
    if scale == 0:
        return data_float, 0

    data_quantized = (data_float / scale * half_level).round()
    quant_scale = 1 / scale * half_level
    if isint == 0:
        data_quantized = data_quantized * scale / half_level
        quant_scale = 1

    return data_quantized, quant_scale


# 给 feature_map 加上 padding
def feature_map_padding(feature_map, padding):
    # feature_map 维度： C, W, H
    while (len(feature_map.shape) < 3):
        feature_map = np.expand_dims(feature_map, axis = 0)
    feature_map_pad = np.pad(feature_map, ((0, 0), (padding, padding), (padding, padding)), mode = 'constant')
    return feature_map_pad


# 将忆阻器每层的输出 out_put 转换回 feature_map 的形式
def output_to_feature_map(out_put, out_w, out_h):
    # out_put shape = [W_out * H_out, C_out]
    # feature_map shape = [C_out, W_out, H_out]
    channels = out_put.shape[1]
    feature_map = out_put.transpose(1, 0).reshape([channels, out_w, out_h])
    return feature_map


# 将 feature_map 转化为下一层忆阻器的输入 array_input
def feature_map_to_input(feature_map, kernel_size, stride, padding, repeat = None):
    # feature_map shape = [C_in, W_in, H_in]
    # array_input shape = [W_out * H_out, C_out]
    while (len(feature_map.shape) < 3):
        feature_map = np.expand_dims(feature_map, axis = 0)
    in_channels = feature_map.shape[0]
    feature_in_w = feature_map.shape[1]
    feature_in_h = feature_map.shape[2]
    feature_out_w = int((feature_in_w - kernel_size + 2 * padding) / stride + 1)
    feature_out_h = int((feature_in_h - kernel_size + 2 * padding) / stride + 1)
    feature_map = feature_map_padding(feature_map, padding)
    input_rows = kernel_size ** 2 * in_channels
    output_rows = feature_out_w * feature_out_h
    array_input = np.zeros([input_rows, output_rows])
    idx = 0
    for i in range(feature_out_w):
        for j in range(feature_out_h):
            slide_window = feature_map[:, i * stride:i * stride + kernel_size,
                           j * stride:j * stride + kernel_size]
            array_input[:, idx] = slide_window.reshape(-1)
            idx += 1
    if repeat:
        array_input = np.tile(array_input, [repeat[0], 1])
    return array_input


# 如果权重复制, 则返回一个复制的 input
def input_repeat(input, row_repeat):
    input_rep = np.tile(input, [row_repeat, 1])
    return input_rep


# 重复的矩阵求出平均值
def weight_avg(weight, repeat):
    # weight 是从忆阻器中读出的多个重复的矩阵
    # repeat 是这个矩阵重复的次数, 格式为 [row_repeat, col_repeat]
    # 实际权重值的大小为 [weight.shape[0] / row_repeat, weight.shape[1] / col_repeat]
    weight_avg = np.zeros([weight.shape[0] // repeat[0], weight.shape[1] // repeat[1]])
    m, n = weight_avg.shape
    for row in range(repeat[0]):
        for col in range(repeat[1]):
            weight_avg += weight[row * m:(row + 1) * m, col * n:(col + 1) * n]
    weight_avg /= (repeat[0] * repeat[1])
    return weight_avg


# 计算网络推理的 softmax, 即概率
def softmax(cross_bar_final_output):
    cross_bar_final_output = cross_bar_final_output.squeeze()
    max_num = cross_bar_final_output.max()
    cross_bar_final_output -= max_num
    prob = np.exp(cross_bar_final_output.max()) / np.exp(cross_bar_final_output).sum()
    return prob


def input_bitwise_expansion_fast(input, dense = True, assign_pulses = None):
    # input 是一个按照 144k 输入数据格式重新排列好的数, 该函数将其每列进行 bitwise 展开
    # input size = [rows, cols]
    # 如果 dense != True:
    #   返回值是一个展开后的稀疏二维数组, 按照 input 中的最大值作为 bitwise 展开的位数, 对每列进行展开
    # output size = [rows, cols * bitlen]
    # 如果 dense == True:
    #   返回值是一个稠密二维数组, 分别按照 input 中每列的最大值作为 bitwise 的展开位数。
    # 即展开后都是 0 的列直接丢弃
    # 在计算时需要知道 bitlen_map 中每个 col 展开了多少次, 根据该值对 144k 的计算结果进行求和处理
    # assign_pulses -> 指定bitwise展开的脉冲次数，如果不是None，则自动判定dense = Fasle

    # 如果输入为全 0 矩阵, 直接返回其本身, 在后续的 mvm 中跳过这次计算
    if (input == 0).all():
        return input, []

    if len(input.shape) == 1:
        input = input.reshape(-1, 1).astype(np.int32)
    input = input.astype(np.int32)
    input_ori = input + 0

    max_range = int(np.max(abs(input)))
    rows = input.shape[0]
    cols = input.shape[1]

    input = input.transpose(1, 0).reshape(-1, 1)

    # 用来存放展开后的矩阵
    # input_expanded 保存了 m 个与 input 相同大小的矩阵(reshape 为 1 列), 作为展开的结果, 每个矩阵元素都是+-1,0。
    input_expanded = np.zeros((rows * cols, max_range), dtype = int)

    # 对 input 矩阵进行并行 bitwise 展开,
    # 当 input 中有任意元素不为零, 则对该元素-1 或+1, 并生成一个与 input 相同大小, 且元素为-1,0,+1 的矩阵, 作为该次展开的值
    # t_expand = time.time()
    for i in range(max_range):
        # bit_cur 为一次并行 bitwise 展开的矩阵, 元素全部为-1,0,+1
        bit_cur = (input > 0) * 1 + (input < 0) * -1
        # 将该次展开的 bit_cur 保存到 input_expanded 中
        input_expanded[:, i:i + 1] = bit_cur
        # 在 input 中-1 或+1, 直到所有值为 0
        input -= bit_cur

    input_expanded = input_expanded.reshape(-1, rows, max_range).transpose(1, 0, 2)
    input_expanded = input_expanded.reshape(rows, -1)

    # 将展开后的稀疏矩阵中的全 0 列删除, 跳过计算
    if dense == True:
        # t = time.time()
        mask = abs(input_expanded).sum(axis = 0)
        input_expanded = input_expanded[:, mask != 0]
        zero_array_num = (mask == 0).astype(int).reshape(1, -1)
        zero_array_num = zero_array_num.reshape(-1, max_range).sum(axis = 1)
        # 获取 input 中, 每一列的最大值, 作为该列 bitwise 展开的位数
        bitlen_map = max_range - zero_array_num

    # 如果指定了每列的展开长度，则对input_expanded补0或截取
    elif assign_pulses:
        expanded_bitlen = input_expanded.shape[1]
        diff = assign_pulses - expanded_bitlen
        if diff > 0:
            zero_mat = np.zeros_like(input_expanded[:, 0])
            input_expanded = np.concatenate([input_expanded, zero_mat], axis = 1)
        else:
            input_expanded = input_expanded[:, 0:assign_pulses]
        bitlen_map = (np.ones([cols]) * assign_pulses).astype(np.int32)

    else:
        bitlen_map = (np.ones([cols]) * abs(input_ori).max()).astype(np.int32)

    return input_expanded, bitlen_map


# bitwise 乘加运算
def mvm_bitwise_concat_push_fast_144k(sdk, input, addr, repeat = None, it_time = 5, verbose = 0):
    # cal_times 是该层运算的总次数，如卷积滑窗的次数，如果是全连接则 cal_times = 1
    cal_times = input.shape[1]
    # 输出通道数
    output_cols = addr[3]
    # 创建一个全零的输出矩阵
    output = np.zeros([cal_times, output_cols])

    # 对输入的二维矩阵 input 做 bitwise 展开, 默认情况下返回一个稠密矩阵
    #   input_expanded 是一个只有 +1,0，-1的矩阵
    #   bitlen_map 中记录了 input 中每一列展开的最大 bit 位数,
    #   在 144k 计算完毕后会根据 bitlen_map 中记录的 bit 位数做对应行的累加
    input_expanded, bitlen_map = input_bitwise_expansion_fast(input)
    output_bitwise = np.array(
        sdk.calculate(input_expanded.transpose(1, 0), addr = addr, it_time = it_time)).astype(np.int8) - 8

    output_bitwise_row = 0
    output_row = 0

    # 对计算结果按照展开的位数进行求和
    for j in bitlen_map:
        if j == 0:
            output[output_row, :] = 0
        else:
            output[output_row, :] = output_bitwise[output_bitwise_row:output_bitwise_row + j].sum(axis = 0)
        output_row += 1
        output_bitwise_row += j

    # ==================================================== #
    # for debug
    if verbose:
        count_max = (output_bitwise >= 6).sum()
        count_min = (output_bitwise <= -6).sum()
        tot_element = np.prod(output_bitwise.shape)
        max_percent = count_max / tot_element * 100
        min_percent = count_min / tot_element * 100
        print(f'output_mean = {output.mean()}')
        print(f'output_shape = {output.shape}')
        print(f'ADC_max_precentage = {max_percent}%')
        print(f'ADC_min_precentage = {min_percent}%')
    # ==================================================== #

    # 如果权重复制了, 求出 output 的平均值
    if repeat:
        row_repeat = repeat[0]
        col_repeat = repeat[1]
        output_avg_cols = int(output_cols / col_repeat)
        output_avg = np.zeros([cal_times, output_avg_cols])
        for i in range(col_repeat):
            output_avg += output[:, i * output_avg_cols: (i + 1) * output_avg_cols]
        output_avg /= col_repeat
        output_avg /= row_repeat
        return output_avg
    return output


# CPU MVM仿真器
def mvm_bitwise_concat_push_fast(input, weight, repeat = None, verbose = 0):
    # bitwise乘加运算
    cal_times = input.shape[1]
    output_cols = weight.shape[1]
    output = np.zeros([cal_times, output_cols])

    # ========== DEBUG ========= #
    time_s = time.time()
    # ========================== #

    input_expanded, bitlen_map = input_bitwise_expansion_fast(input)

    # ========== DEBUG ========= #
    time_e = time.time()
    time_elapsed = time_e - time_s
    if verbose:
        print(f'bitwise expansion time = {time_elapsed}')
    # ========================== #

    # 如果feature map展开后的返回值为全0矩阵(即feature map)为全0, 直接跳过这次运算, 返回一个尺寸正确的全0矩阵作为结果
    if (input_expanded == 0).all():
        if repeat:
            col_repeat = repeat[1]
            return output[:, 0:int(output_cols / col_repeat)]
        else:
            return output
    output_bitwise = sdk_cal_sim(input_expanded, weight)

    output_bitwise_col = 0
    output_row = 0

    # ========== DEBUG ========= #
    time_s = time.time()
    # ========================== #

    # 对计算结果按照展开的位数进行求和
    for j in bitlen_map:
        if j == 0:
            output[output_row] = 0
        else:
            output[output_row] = output_bitwise[output_bitwise_col:output_bitwise_col + j].sum(axis = 0)
        output_row += 1
        output_bitwise_col += j

    # 如果权重复制了, 求出output的平均值
    if repeat:
        row_repeat = repeat[0]
        col_repeat = repeat[1]
        cols_output_avg = int(output_cols / col_repeat)
        output_avg = np.zeros([cal_times, cols_output_avg])
        for i in range(col_repeat):
            output_avg += output[:, i * cols_output_avg: (i + 1) * cols_output_avg]
        output_avg /= col_repeat
        output_avg /= row_repeat

        # ========== DEBUG ========= #
        time_e = time.time()
        time_elapsed = time_e - time_s
        if verbose:
            print(f'bitwise expansion time = {time_elapsed}')
        # ========================== #

        return output_avg

    # ========== DEBUG ========= #
    time_e = time.time()
    time_elapsed = time_e - time_s
    if verbose:
        print(f'bitwise expansion time = {time_elapsed}')
    # ========================== #
    return output


# simulate the mvm process in C200 SDK
def sdk_cal_sim(input, weight, it_time = 1):
    cal_times = input.shape[1]
    sum = []
    for i in range(cal_times):
        temp = (input[:, i].reshape(-1, 1) * weight).sum(axis = 0)
        sum.append(temp.reshape(1, -1))
    result = np.concatenate(sum, axis = 0)
    result *= it_time
    return result

# =========================================================== #
# Discarded Functions
# =========================================================== #

# 144k 片上运算的函数
# def mvm_bitwise_144k(sdk, input, addr, repeat = None, it_time = 5, verbose = 0):
#     # bitwise 乘加运算
#     cal_times = input.shape[1]
#     output_cols = addr[3]
#     # print(f'output_cols = {output_cols}')
#     output = np.zeros([cal_times, output_cols])

#     # for debug:
#     count_max = 0
#     count_min = 0
#     tot_element = 0
#     for i in range(cal_times):
#         col = input[:, i]
#         bit_len = int(abs(col).max())
#         col_expanded = col_bitwise_expansion(col, bit_len)
#         temp = 0

#         for j in range(bit_len):
#             time_cal_for_1_bit = time.time()
#             ADC_out = np.array(sdk.calculate(col_expanded[:,j], addr = addr, it_time = it_time)).astype(np.int8) - 8

#             # for debug:
#             count_max += (ADC_out >= 6).sum()
#             count_min += (ADC_out <= -6).sum()
#             tot_element += np.prod(ADC_out.shape)


#             temp += ADC_out
#             time_cal_for_1_bit = time.time() - time_cal_for_1_bit
#             # print(f'time_cal_for_1_bit = {time_cal_for_1_bit}')

#             # if verbose:
#             #     print(f'ADC_out = {ADC_out}')
#             #     print(f'max = {ADC_out.max()}')
#             #     print(f'min = {ADC_out.min()}')
#         output[i, :] = temp
#     print(f'output_mean = {output.mean()}')
#     print(f'output_shape = {output.shape}')
#     if verbose:
#             max_percent = count_max / tot_element * 100
#             min_percent = count_min / tot_element * 100
#             print(f'ADC_max_precentage = {max_percent}%')
#             print(f'ADC_min_precentage = {min_percent}%')

#     # 如果权重复制了, 求出 output 的平均值
#     if repeat:
#         row_repeat = repeat[0]
#         col_repeat = repeat[1]
#         cols_output_avg = int(output_cols / col_repeat)
#         output_avg = np.zeros([cal_times, cols_output_avg])
#         for i in range(col_repeat):
#             output_avg += output[:, i * cols_output_avg: (i + 1) * cols_output_avg]
#         output_avg /= col_repeat
#         output_avg /= row_repeat
#         return output_avg
#     return output


# def mvm_bitwise_concat_push_144k(sdk, input, addr, repeat = None, it_time = 5, verbose = 0):
#     # bitwise 乘加运算
#     cal_times = input.shape[1]
#     output_cols = addr[3]
#     output = np.zeros([cal_times, output_cols])
#     bitlen_map = []
#     input_expanded = []

#     time_expansion = time.time()
#     for i in range(cal_times):
#         col = input[:, i]
#         bit_len = int(abs(col).max())
#         # log down bit length for current column
#         bitlen_map.append(bit_len)
#         # column bit-wise expansion
#         col_expanded = col_bitwise_expansion(col, bit_len)
#         input_expanded.append(col_expanded)
#     time_expansion = time.time() - time_expansion
#     # print(f'time_expansion = {time_expansion}')

#     input_expanded = np.concatenate(input_expanded, axis = 1)

#     time_cal = time.time()
#     out_put_bitwise = np.array(sdk.calculate(input_expanded.transpose(1,0), addr = addr, it_time = it_time)).astype(np.int8) - 8
#     # print(out_put_bitwise.shape)
#     # print(out_put_bitwise[0])
#     # print(out_put_bitwise[1])
#     time_cal = time.time() - time_cal
#     # print(f'On chip cal time = {time_cal}')
#     count_max = (out_put_bitwise >= 6).sum()
#     count_min = (out_put_bitwise <= -6).sum()
#     tot_element = np.prod(out_put_bitwise.shape)
#     max_percent = count_max / tot_element * 100
#     min_percent = count_min / tot_element * 100
#     if verbose:
#         print(f'ADC_max_precentage = {max_percent}%')
#         print(f'ADC_min_precentage = {min_percent}%')

#     out_put_bitwise_col = 0
#     output_row = 0
#     for j in bitlen_map:
#         if j == 0:
#             j = 1
#         output[output_row] = out_put_bitwise[out_put_bitwise_col:out_put_bitwise_col + j].sum(axis = 0)
#         output_row += 1
#         out_put_bitwise_col += j

#     # 如果权重复制了, 求出 output 的平均值
#     if repeat:
#         row_repeat = repeat[0]
#         col_repeat = repeat[1]
#         cols_output_avg = int(output_cols / col_repeat)
#         output_avg = np.zeros([cal_times, cols_output_avg])
#         for i in range(col_repeat):
#             output_avg += output[:, i * cols_output_avg: (i + 1) * cols_output_avg]
#         output_avg /= col_repeat
#         output_avg /= row_repeat
#         return output_avg.astype(np.int32)
#     return output.astype(np.int32)
