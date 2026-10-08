// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <riscv_vector.h>

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/general_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/conv_2d_nchw_fchw_riscv_64_internal.h"

void iree_uk_conv_tile_generic_riscv_64_direct(
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
      size_t vl = __riscv_vsetvl_e32m1(filter_size_w);
      for (iree_uk_index_t kw = 0; kw < filter_size_w; kw += vl) {
        vl = __riscv_vsetvl_e32m1(filter_size_w - kw);
        iree_uk_index_t ih = oh * stride_h + kh;
        iree_uk_index_t iw = ow * stride_w + kw;
        if (ih >= 0 && ih < in_size_h &&
            iw >= 0 && iw < in_size_w) {
          iree_uk_index_t in_idx = in_offset +
                                  n * in_stride0 +
                                  ic * in_stride1 +
                                  ih * in_stride2 +
                                  iw;
          iree_uk_index_t filter_idx = filter_offset +
                                      oc * filter_stride0 +
                                      ic * filter_stride1 +
                                      kh * filter_size_w +
                                      kw;
          //float* in_ptr = (float*)((char*)in_tile_ptr + in_idx * in_type_size);
          //float* filter_ptr = (float*)((char*)filter_tile_ptr + filter_idx * filter_type_size);
          float* in_ptr = (float*)((char*)in_tile_ptr + 0);
          float* filter_ptr = (float*)((char*)filter_tile_ptr + 0);
          //sum += (*in_ptr) * (*filter_ptr);
          vfloat32m1_t v0 = __riscv_vle32_v_f32m1(&in_ptr[in_idx], vl);
          vfloat32m1_t v1 = __riscv_vle32_v_f32m1(&filter_ptr[filter_idx], vl);
          vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(v0, v1, vl);
          sum += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(vprod, __riscv_vfmv_s_f_f32m1(0.0, vl), vl));
        }
      }
    }
  }
  *out_ptr = sum;
}

void iree_uk_conv_2d_nchw_fchw_generic_tile_riscv_64(
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
          size_t vl = __riscv_vsetvl_e32m1(filter_size_w);
          for (iree_uk_index_t kw = 0; kw < filter_size_w; kw+=vl) {
            vl = __riscv_vsetvl_e32m1(filter_size_w - kw);
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
              //float* in_ptr = (float*)((char*)in_tile_ptr + in_idx * in_type_size);
              //float* filter_ptr = (float*)((char*)filter_tile_ptr + filter_idx * filter_type_size);
              //sum += (*in_ptr) * (*filter_ptr);
              float* in_ptr = (float*)((char*)in_tile_ptr + 0);
              float* filter_ptr = (float*)((char*)filter_tile_ptr + 0);
              vfloat32m1_t v0 = __riscv_vle32_v_f32m1(&in_ptr[in_idx], vl);
              vfloat32m1_t v1 = __riscv_vle32_v_f32m1(&filter_ptr[filter_idx], vl);
              vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(v0, v1, vl);
              sum += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(vprod, __riscv_vfmv_s_f_f32m1(0.0, vl), vl));
            }
          }
        }
      }
      *out_ptr = sum;
    }
  }
}

void iree_uk_conv_2d_nchw_fchw_generic_tile_riscv_64_pack(
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
	  size_t vl;
          for (iree_uk_index_t kw = 0; kw < filter_size_w; kw += vl) {
            vl = __riscv_vsetvl_e32m1(filter_size_w - kw);
            vfloat32m1_t v0 = __riscv_vle32_v_f32m1(&in_row[kw], vl);
            vfloat32m1_t v1 = __riscv_vle32_v_f32m1(&filter_row[kw], vl);
            vfloat32m1_t vprod = __riscv_vfmul_vv_f32m1(v0, v1, vl);
            sum += __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m1_f32m1(vprod, __riscv_vfmv_s_f_f32m1(0.0f, vl), vl));
          }
        }
      }
      *out_ptr = sum;
    }
  }
}
