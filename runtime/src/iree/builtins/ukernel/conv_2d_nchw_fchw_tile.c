// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/builtins/ukernel/conv_2d_nchw_fchw_internal.h"

static void iree_uk_conv_tile_generic_direct(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, const void* IREE_UK_RESTRICT filter_tile_ptr,
    iree_uk_index_t in_size_c, iree_uk_index_t out_size_c,
    iree_uk_index_t n, iree_uk_index_t oc,
    iree_uk_index_t oh, iree_uk_index_t ow,
    iree_uk_index_t in_size_h, iree_uk_index_t in_size_w,
    iree_uk_index_t filter_size_h, iree_uk_index_t filter_size_w,
    iree_uk_index_t out_size_h, iree_uk_index_t out_size_w,
    iree_uk_index_t tile_size0, iree_uk_index_t tile_size1,
    iree_uk_index_t in_stride0, iree_uk_index_t in_stride1, iree_uk_index_t in_stride2,
    iree_uk_index_t filter_stride0, iree_uk_index_t filter_stride1,
    iree_uk_index_t out_stride0, iree_uk_index_t out_stride1,
    iree_uk_index_t in_offset, iree_uk_index_t filter_offset,
    iree_uk_index_t in_type_size, iree_uk_index_t filter_type_size,
    iree_uk_index_t out_type_size, iree_uk_uint32_t flags) {
  float* out_ptr = (float*)((char*)out_tile_ptr + 0);
  float sum = (flags & IREE_UK_FLAG_CONV_ACCUMULATE) ? (*out_ptr) : 0.0f;
  iree_uk_index_t stride_h = (((in_size_h-filter_size_h)/1+1 == out_size_h) ? 1 :
                            (((in_size_h-filter_size_h)/2+1 == out_size_h) ? 2 : 4));
  iree_uk_index_t stride_w = (((in_size_w-filter_size_w)/1+1 == out_size_w) ? 1 :
                            (((in_size_w-filter_size_w)/2+1 == out_size_w) ? 2 : 4));
  for (iree_uk_index_t ic = 0; ic < in_size_c; ic++) {
    for (iree_uk_index_t kh = 0; kh < filter_size_h; kh++) {
      for (iree_uk_index_t kw = 0; kw < filter_size_w; kw++) {
        iree_uk_index_t ih = oh * stride_h + kh;
        iree_uk_index_t iw = ow * stride_w + kw;
        if (ih >= 0 && ih < in_size_h &&
            iw >= 0 && iw < in_size_w) {
          iree_uk_index_t in_idx = in_offset +
                                  n * in_stride0 +
                                  ic * in_stride1 +
                                  ih * in_stride2 +
                                  iw;
          iree_uk_index_t filter_idx = filter_offset+
                                      oc * filter_stride0 +
                                      ic * filter_stride1 +
                                      kh * filter_size_w +
                                      kw;
          float* in_ptr = (float*)((char*)in_tile_ptr + in_idx * in_type_size);
          float* filter_ptr = (float*)((char*)filter_tile_ptr + filter_idx * filter_type_size);
          sum += (*in_ptr) * (*filter_ptr);
        }
      }
    }
  }
  *out_ptr = sum;
}

static void iree_uk_conv_2d_nchw_fchw_generic_tile(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr, const void* IREE_UK_RESTRICT filter_tile_ptr,
    iree_uk_index_t in_size_c, iree_uk_index_t out_size_c,
    iree_uk_index_t n, iree_uk_index_t oc,
    iree_uk_index_t oh, iree_uk_index_t ow,
    iree_uk_index_t in_size_h, iree_uk_index_t in_size_w,
    iree_uk_index_t filter_size_h, iree_uk_index_t filter_size_w,
    iree_uk_index_t out_size_h, iree_uk_index_t out_size_w,
    iree_uk_index_t tile_size0, iree_uk_index_t tile_size1,
    iree_uk_index_t in_stride0, iree_uk_index_t in_stride1, iree_uk_index_t in_stride2,
    iree_uk_index_t filter_stride0, iree_uk_index_t filter_stride1,
    iree_uk_index_t out_stride0, iree_uk_index_t out_stride1,
    iree_uk_index_t in_offset, iree_uk_index_t filter_offset,
    iree_uk_index_t in_type_size, iree_uk_index_t filter_type_size,
    iree_uk_index_t out_type_size, iree_uk_uint32_t flags) {
  iree_uk_index_t stride_h = (((in_size_h-filter_size_h)/1+1 == out_size_h) ? 1 :
                            (((in_size_h-filter_size_h)/2+1 == out_size_h) ? 2 : 4));
  iree_uk_index_t stride_w = (((in_size_w-filter_size_w)/1+1 == out_size_w) ? 1 :
                            (((in_size_w-filter_size_w)/2+1 == out_size_w) ? 2 : 4));
  for (iree_uk_index_t i = 0; i < tile_size0; i++) {
    for (iree_uk_index_t j = 0; j < tile_size1; j++) {
      if (oh + i >= out_size_h || ow + j >= out_size_w) continue;
      float* out_ptr = (float*)((char*)out_tile_ptr + (i * out_size_w + j) * out_type_size);
      float sum = (flags & IREE_UK_FLAG_CONV_ACCUMULATE) ? (*out_ptr) : 0.0f;
      for (iree_uk_index_t ic = 0; ic < in_size_c; ic++) {
        for (iree_uk_index_t kh = 0; kh < filter_size_h; kh++) {
          for (iree_uk_index_t kw = 0; kw < filter_size_w; kw++) {
            iree_uk_index_t ih = (oh + i) * stride_h + kh;
            iree_uk_index_t iw = (ow + j) * stride_w + kw;
            if (ih >= 0 && ih < in_size_h &&
                iw >= 0 && iw < in_size_w) {
              iree_uk_index_t in_idx = in_offset +
                                      n * in_stride0 +
                                      ic * in_stride1 +
                                      ih * in_stride2 +
                                      iw;
              iree_uk_index_t filter_idx = filter_offset+
                                          oc * filter_stride0 +
                                          ic * filter_stride1 +
                                          kh * filter_size_w +
                                          kw;
              float* in_ptr = (float*)((char*)in_tile_ptr + in_idx * in_type_size);
              float* filter_ptr = (float*)((char*)filter_tile_ptr + filter_idx * filter_type_size);
              sum += (*in_ptr) * (*filter_ptr);
            }
          }
        }
      }
      *out_ptr = sum;
    }
  }
}

static void iree_uk_conv_2d_nchw_fchw_generic_tile_pack(
    void* IREE_UK_RESTRICT out_tile_ptr,
    const void* IREE_UK_RESTRICT in_tile_ptr,
    const void* IREE_UK_RESTRICT filter_tile_ptr,
    iree_uk_index_t in_size_c, iree_uk_index_t out_size_c,
    iree_uk_index_t n, iree_uk_index_t oc,
    iree_uk_index_t oh, iree_uk_index_t ow,
    iree_uk_index_t in_size_h, iree_uk_index_t in_size_w,
    iree_uk_index_t filter_size_h, iree_uk_index_t filter_size_w,
    iree_uk_index_t out_size_h, iree_uk_index_t out_size_w,
    iree_uk_index_t tile_size0, iree_uk_index_t tile_size1,
    iree_uk_index_t in_stride0, iree_uk_index_t in_stride1, iree_uk_index_t in_stride2,
    iree_uk_index_t filter_stride0, iree_uk_index_t filter_stride1,
    iree_uk_index_t out_stride0, iree_uk_index_t out_stride1,
    iree_uk_index_t in_offset, iree_uk_index_t filter_offset,
    iree_uk_index_t in_type_size, iree_uk_index_t filter_type_size,
    iree_uk_index_t out_type_size, iree_uk_uint32_t flags) {
  iree_uk_index_t stride_h = (((in_size_h-filter_size_h)/1+1 == out_size_h) ? 1 :
                            (((in_size_h-filter_size_h)/2+1 == out_size_h) ? 2 : 4));
  iree_uk_index_t stride_w = (((in_size_w-filter_size_w)/1+1 == out_size_w) ? 1 :
                            (((in_size_w-filter_size_w)/2+1 == out_size_w) ? 2 : 4));
  const float* in_base = (const float*)in_tile_ptr + in_offset + n * in_stride0;
  const float* filter_base = (const float*)filter_tile_ptr + filter_offset + oc * filter_stride0;
  for (iree_uk_index_t i = 0; i < tile_size0; i++) {
    for (iree_uk_index_t j = 0; j < tile_size1; j++) {
      if (oh + i >= out_size_h || ow + j >= out_size_w) continue;
      float* out_ptr = (float*)((char*)out_tile_ptr + (i * out_stride1 + j) * out_type_size);
      float sum = (flags & IREE_UK_FLAG_CONV_ACCUMULATE) ? (*out_ptr) : 0.0f;
      iree_uk_index_t current_oh = oh + i;
      iree_uk_index_t current_ow = ow + j;
      iree_uk_index_t start_iw = current_ow * stride_w;
      for (iree_uk_index_t ic = 0; ic < in_size_c; ic++) {
        const float* in_channel_base = in_base + ic * in_stride1;
        const float* filter_channel_base = filter_base + ic * filter_stride1;
        for (iree_uk_index_t kh = 0; kh < filter_size_h; kh++) {
          iree_uk_index_t ih = current_oh * stride_h + kh;
          const float* in_row = in_channel_base + ih * in_stride2 + start_iw;
          const float* filter_row = filter_channel_base + kh * filter_size_w;
          for (iree_uk_index_t kw = 0; kw < filter_size_w; kw++) {
            sum += in_row[kw] * filter_row[kw];
          }
        }
      }
      *out_ptr = sum;
    }
  }
}

iree_uk_conv_2d_nchw_fchw_tile_func_t iree_uk_conv_2d_nchw_fchw_select_tile_func(
    const iree_uk_conv_params_t* params) {
  iree_uk_conv_2d_nchw_fchw_tile_func_t arch_tile_func =
      iree_uk_conv_2d_nchw_fchw_select_tile_func_arch(params);
  if (arch_tile_func) return arch_tile_func;
  //return iree_uk_conv_tile_generic_direct;
  return iree_uk_conv_2d_nchw_fchw_generic_tile;
}
