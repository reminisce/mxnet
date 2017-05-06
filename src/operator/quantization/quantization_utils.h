/*!
 *  Copyright (c) 2017 by Contributors
 * \file quantization_utils-inl.h
 * \brief (TODO)
 */
#ifndef MXNET_OPERATOR_QUANTIZATION_UTILS_H_
#define MXNET_OPERATOR_QUANTIZATION_UTILS_H_

#include <mxnet/base.h>
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

using mshadow::red::limits::MinValue;
using mshadow::red::limits::MaxValue;

template<typename T>
MSHADOW_XINLINE int64_t FloatToQuantizedUnclamped(
    float input, float min_range, float max_range) {
  const int64_t lowest_quantized = static_cast<double>(MinValue<T>());
  if (min_range == max_range) return lowest_quantized;
  const int num_of_bits = sizeof(T) * 8;
  const int64_t num_of_steps = static_cast<int64_t>(1) << num_of_bits;
  const double range_adjust = (num_of_steps / (num_of_steps - 1.0));
  const double range = ((max_range - min_range) * range_adjust);
  const double range_scale = (num_of_steps / range);
  int64_t quantized =
    (round(input * range_scale) - round(min_range * range_scale));
  quantized += lowest_quantized;
  return quantized;
}

template<typename T>
MSHADOW_XINLINE T FloatToQuantized(
    float input, float min_range, float max_range) {
  int64_t quantized = FloatToQuantizedUnclamped<T>(input, min_range, max_range);
  const int64_t lowest_quantized  = static_cast<double>(MinValue<T>());
  const int64_t highest_quantized = static_cast<double>(MaxValue<T>());
  quantized = std::max(quantized, lowest_quantized);
  quantized = std::min(quantized, highest_quantized);
  return static_cast<T>(static_cast<int32_t>(quantized));
}

template <typename T>
MSHADOW_XINLINE float QuantizedToFloat(
    float input, float range_min, float range_max) {
  if (range_min == range_max) {
    return range_min;
  }
  const int number_of_bits = sizeof(T) * 8;
  const int64_t number_of_steps = static_cast<int64_t>(1) << number_of_bits;
  const double range_adjust = (number_of_steps / (number_of_steps - 1.0));
  const double range = ((range_max - range_min) * range_adjust);
  const double range_scale = (range / number_of_steps);
  const int64_t lowest_quantized =
      static_cast<int64_t>(MinValue<T>());
  const double offset_input = static_cast<double>(input) - lowest_quantized;
  // For compatibility with DEQUANTIZE_WITH_EIGEN, we should convert
  // range_scale to a float, otherwise range_min_rounded might be slighly
  // different.
  const double range_min_rounded =
      round(range_min / static_cast<float>(range_scale)) *
      static_cast<float>(range_scale);
  const double result = range_min_rounded + (offset_input * range_scale);
  return static_cast<float>(result);
}

template <class T1, class T2>
MSHADOW_XINLINE T2 RequantizeInNewRange(T1 input, float min_input, float max_input,
                               float min_new, float max_new) {
  const float input_float = QuantizedToFloat<T1>(input, min_input, max_input);
  return FloatToQuantized<T2>(input_float, min_new, max_new);
}

template <class T1, class T2>
MSHADOW_XINLINE void RequantizeManyInNewRange(size_t count,
	T2* output, const T1 *input, float input_min,
	float input_max, float actual_min, float actual_max) {

  for (size_t index = 0; index < count; ++index) {
    const float input_float =
        QuantizedToFloat<T1>(input[index], input_min, input_max);
    output[index] = FloatToQuantized<T2>(input_float, actual_min, actual_max);
  }
}

/*

// Because converting 32-bit accumulated results down to eight bit is a common
// case, we have a specialized code path to handle it as efficiently as
// possible using only fixed-point math for the inner loop.
template <>
inline void RequantizeManyInNewRange<qint32, quint8>(
    const qint32* input, size_t count, float min_input, float max_input,
    float min_output, float max_output, quint8* output) {
  // Initially we calculate all the constants we need once, before we go into
  // the inner loop.  If this is updated, also update the Eigen version.
  const int fp_shift = 16;
  const float input_range = max_input - min_input;
  const float output_range = max_output - min_output;
  const float recip_output_range =
      output_range == 0.0 ? 0.0 : (255.0 / output_range);
  const float input_rezero = (min_input + max_input) / 2.0;
  const int64 range_scale_fp =
      output_range == 0.0 ? 0.0
                          : static_cast<int64>(255.0 * (1 << fp_shift) *
                                               input_range / output_range);
  const int64 input_offset_fp =
      static_cast<int64>(input_rezero * recip_output_range * (1 << fp_shift));
  const int64 output_offset_fp =
      output_range == 0.0 ? 0 : static_cast<int64>((1 << fp_shift) *
                                                   (min_output * 255.0) /
                                                   output_range);
  const int64 rounding_delta = 1 << (fp_shift - 1);

  // Inside this loop we just do minimal adds, multiplies, and shifts, in a way
  // that could be easily adapted for a SIMD implementation. It should also be
  // possible to perform all the calculations in 32-bit rather than 64, but
  // that's not been implemented yet.
  for (size_t index = 0; index < count; ++index) {
    const int64 input_value = static_cast<int64>(input[index]);
    const int64 fp_value =
        ((input_value * range_scale_fp) >> 32) + input_offset_fp;
    const int64 offset_intermediate = fp_value - output_offset_fp;
    const int64 round_intermediate = offset_intermediate + rounding_delta;
    int64 quantized_int64 = round_intermediate >> fp_shift;
    quantized_int64 = std::max(quantized_int64, 0LL);
    quantized_int64 = std::min(quantized_int64, 255LL);
    output[index] = static_cast<quint8>(static_cast<int32>(quantized_int64));
  }
}

*/


template<typename T>
MSHADOW_XINLINE float FloatForOneQuantizedLevel(
    float range_min, float range_max) {
  const int64_t highest = static_cast<int64_t>(MaxValue<T>());
  const int64_t lowest  = static_cast<int64_t>(MinValue<T>());
  const float float_for_one_quantized_level =
      (range_max - range_min) / (highest - lowest);
  return float_for_one_quantized_level;
}

template <typename TA, typename TB, typename TC>
MSHADOW_XINLINE void QuantizationRangeForMultiplication(
    float min_a, float max_a, float min_b, float max_b,
    float* min_c, float* max_c) {
  const float a_float_for_one_quant_level =
    FloatForOneQuantizedLevel<TA>(min_a, max_a);
  const float b_float_for_one_quant_level =
    FloatForOneQuantizedLevel<TB>(min_b, max_b);

  const int64_t c_highest =
    static_cast<int64_t>(MaxValue<TC>());
  const int64_t c_lowest  =
    static_cast<int64_t>(MinValue<TC>());
  const float c_float_for_one_quant_level =
    a_float_for_one_quant_level * b_float_for_one_quant_level;

  *min_c = c_float_for_one_quant_level * c_lowest;
  *max_c = c_float_for_one_quant_level * c_highest;
}

struct QuantizationRangeForMultiplicationStruct {
  MSHADOW_XINLINE static void Map(int i,
                                  float *min_c,
                                  float *max_c,
                                  const float *min_a,
                                  const float *max_a,
                                  const float *min_b,
                                  const float *max_b) {
  QuantizationRangeForMultiplication<int8_t, int8_t, int32_t>(
    min_a[i], max_a[i], min_b[i], max_b[i], min_c, max_c);
  }
};



// This is an unoptimized but debuggable implementation of the GEMM matrix
// multiply function, used to compare to faster but more opaque versions, or
// for bit depths or argument combinations that aren't supported by optimized
// code.
// It assumes the row-major convention used by MXNet, and implements
// C = A * B, like the standard BLAS GEMM interface. If the tranpose flags are
// true, then the relevant matrix is treated as stored in column-major order.

template <class T1, class T2, class T3>
void ReferenceGemm(bool transpose_a, bool transpose_b, bool transpose_c,
                   size_t m, size_t n, size_t k, const T1* a, int32_t offset_a,
                   size_t lda, const T2* b, int32_t offset_b, size_t ldb, T3* c,
                   int32_t shift_c, int32_t offset_c, int32_t mult_c, size_t ldc) {
  int a_i_stride;
  int a_l_stride;
  if (transpose_a) {
    a_i_stride = 1;
    a_l_stride = lda;
  } else {
    a_i_stride = lda;
    a_l_stride = 1;
  }
  int b_j_stride;
  int b_l_stride;
  if (transpose_b) {
    b_j_stride = ldb;
    b_l_stride = 1;
  } else {
    b_j_stride = 1;
    b_l_stride = ldb;
  }
  int c_i_stride;
  int c_j_stride;
  if (transpose_c) {
    c_i_stride = 1;
    c_j_stride = ldc;
  } else {
    c_i_stride = ldc;
    c_j_stride = 1;
  }

  const int32_t highest =
    static_cast<int32_t>(std::numeric_limits<T3>::max());
  const int32_t lowest  =
    static_cast<int32_t>(std::numeric_limits<T3>::min());
  const int32_t rounding =
    (shift_c < 1) ? 0 : (1 << (shift_c - 1));

  int i, j, l;
  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      int32_t total = 0;
      for (l = 0; l < k; l++) {
        const size_t a_index = ((i * a_i_stride) + (l * a_l_stride));
        const int32_t a_value = static_cast<int32_t>(a[a_index]) - offset_a;
        const size_t b_index = ((j * b_j_stride) + (l * b_l_stride));
        const int32_t b_value = static_cast<int32_t>(b[b_index]) - offset_b;
        total += (a_value * b_value);
      }
      const size_t c_index = ((i * c_i_stride) + (j * c_j_stride));
      int32_t output = ((((total + offset_c) * mult_c) + rounding) >> shift_c);
      if (output > highest) {
        output = highest;
      }
      if (output < lowest) {
        output = lowest;
      }
      c[c_index] = static_cast<T3>(output);
    }
  }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZATION_UTILS_H_
