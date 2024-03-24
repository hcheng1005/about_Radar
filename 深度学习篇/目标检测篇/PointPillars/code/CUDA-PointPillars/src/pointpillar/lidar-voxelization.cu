/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_fp16.h>
#include "lidar-voxelization.hpp"

#include "common/check.hpp"
#include "common/launch.cuh"

namespace pointpillar
{
  namespace lidar
  {

    const int POINTS_PER_VOXEL = 32;
    const int WARP_SIZE = 32;
    const int WARPS_PER_BLOCK = 4;
    const int FEATURES_SIZE = 10;

    /*
      这是一个CUDA内核函数,名为generateVoxels_random_kernel,它负责从一组输入点生成体素(voxel)。下面是对函数功能的中文解释:

      函数接受以下参数:

      points: 输入点的内存指针
      points_size: 输入点的数量
      min_x_range, max_x_range, min_y_range, max_y_range, min_z_range, max_z_range: x、y、z坐标的最小和最大范围
      pillar_x_size, pillar_y_size, pillar_z_size: 每个体素在x、y、z维度上的大小
      grid_y_size, grid_x_size: 网格在y和x维度上的大小
      mask: 一个掩码数组指针,用于跟踪每个体素中的点数
      voxels: 输出体素数组的指针
      该函数并行执行,每个线程处理一个输入点。

      对于每个输入点,函数首先检查点是否在指定范围内。如果不在,函数直接返回,不处理该点。

      如果点在范围内,函数根据点的x和y坐标以及体素大小,计算该点所属的体素索引。

      函数然后使用原子操作(atomicAdd)来递增对应体素在mask数组中的点计数。

      如果体素的点计数小于最大点数(POINTS_PER_VOXEL),函数使用原子操作(atomicExch)将点的x、y、z、w坐标存储在voxels数组中。

    */
    /**
     * @description: 
     * @return {*}
     */
    static __global__ void generateVoxels_random_kernel(const float *points, size_t points_size,
                                                        float min_x_range, float max_x_range,
                                                        float min_y_range, float max_y_range,
                                                        float min_z_range, float max_z_range,
                                                        float pillar_x_size, float pillar_y_size, float pillar_z_size,
                                                        int grid_y_size, int grid_x_size,
                                                        unsigned int *mask, float *voxels)
    {
      int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (point_idx >= points_size)
        return;

      float4 point = ((float4 *)points)[point_idx];

      // 点云范围筛选
      if (point.x < min_x_range || point.x >= max_x_range || point.y < min_y_range || point.y >= max_y_range || point.z < min_z_range || point.z >= max_z_range)
        return;

      int voxel_idx = floorf((point.x - min_x_range) / pillar_x_size);
      int voxel_idy = floorf((point.y - min_y_range) / pillar_y_size);
      unsigned int voxel_index = voxel_idy * grid_x_size + voxel_idx;

      /*
        这行代码使用了CUDA提供的原子操作atomicAdd。它的作用是:

        获取当前体素索引voxel_index对应的掩码数组mask中的值。
        对该值进行原子加操作,加 1。
        将加 1 后的新值赋给变量point_id。
        这样做的目的是为了给当前点分配一个唯一的ID,该ID表示该点是当前体素中的第几个点。

        使用原子操作是为了确保在多个线程并发访问同一个体素时,不会出现数据竞争的问题。atomicAdd能够保证对mask数组的读改写操作是原子性的,即一次性完成,不会被其他线程打断。
      */
      unsigned int point_id = atomicAdd(&(mask[voxel_index]), 1);

      if (point_id >= POINTS_PER_VOXEL)
        return;


      /*
        首先,这段代码计算了一个地址指针address,它指向voxels数组中当前体素(voxel_index)对应的存储位置。

        voxel_index * POINTS_PER_VOXEL计算了当前体素在voxels数组中的起始位置。
        point_id是当前点在该体素中的序号。
        乘以4是因为每个点由4个float值(x, y, z, w)组成。
        接下来,代码使用CUDA提供的原子操作atomicExch将当前点的坐标(x, y, z, w)依次写入到计算出的地址address中。

        atomicExch是一个原子交换操作,它可以确保多个线程并发访问同一个内存地址时不会出现数据竞争问题。
        这样可以确保每个点的坐标信息都能被正确地写入到对应的体素中。
        总的来说,这段代码的作用是将当前点的坐标信息以原子方式写入到voxels数组的正确位置,以确保数据的正确性和并发安全性。这是生成体素数据的关键步骤。
      */
      float *address = voxels + (voxel_index * POINTS_PER_VOXEL + point_id) * 4;
      atomicExch(address + 0, point.x);
      atomicExch(address + 1, point.y);
      atomicExch(address + 2, point.z);
      atomicExch(address + 3, point.w);
    }

    /**
     * @description: 
     * @return {*}
     */
    cudaError_t generateVoxels_random_launch(const float *points, size_t points_size,
                                             float min_x_range, float max_x_range,
                                             float min_y_range, float max_y_range,
                                             float min_z_range, float max_z_range,
                                             float pillar_x_size, float pillar_y_size, float pillar_z_size,
                                             int grid_y_size, int grid_x_size,
                                             unsigned int *mask, float *voxels,
                                             cudaStream_t stream)
    {
      dim3 blocks((points_size + 256 - 1) / 256);
      dim3 threads(256);
      generateVoxels_random_kernel<<<blocks, threads, 0, stream>>>(points, points_size,
                                                                   min_x_range, max_x_range,
                                                                   min_y_range, max_y_range,
                                                                   min_z_range, max_z_range,
                                                                   pillar_x_size, pillar_y_size, pillar_z_size,
                                                                   grid_y_size, grid_x_size,
                                                                   mask, voxels);
      cudaError_t err = cudaGetLastError();
      return err;
    }

    static __global__ void generateBaseFeatures_kernel(unsigned int *mask, float *voxels,
                                                       int grid_y_size, int grid_x_size,
                                                       unsigned int *pillar_num,
                                                       float *voxel_features,
                                                       unsigned int *voxel_num,
                                                       unsigned int *voxel_idxs)
    {
      unsigned int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
      unsigned int voxel_idy = blockIdx.y * blockDim.y + threadIdx.y;

      if (voxel_idx >= grid_x_size || voxel_idy >= grid_y_size)
          return;

      unsigned int voxel_index = voxel_idy * grid_x_size + voxel_idx;
      unsigned int count = mask[voxel_index];

      // 判断是否为空pillars
      // 这里的注释说明,函数首先要判断当前处理的体素是否为空体素。
      // 如果体素为空,则直接返回,不需要进行后续的特征提取。
      if (!(count > 0))
          return;
      count = count < POINTS_PER_VOXEL ? count : POINTS_PER_VOXEL;

      unsigned int current_pillarId = 0;
      // 使用原子操作为当前体素分配一个唯一的ID
      // 这个ID将用于后续特征的存储
      current_pillarId = atomicAdd(pillar_num, 1); // pillar_num非空pillar个数++

      voxel_num[current_pillarId] = count;

      uint4 idx = {0, 0, voxel_idy, voxel_idx};
      // 将体素的坐标信息存储到voxel_idxs数组中
      ((uint4 *)voxel_idxs)[current_pillarId] = idx;

      // 遍历该pillar的所有点云
      for (int i = 0; i < count; i++)
      {
          int inIndex = voxel_index * POINTS_PER_VOXEL + i; // 这是每个grid中点云的的索引
          int outIndex = current_pillarId * POINTS_PER_VOXEL + i; // 这是pillar的索引位置
          // 将该体素中所有点的特征信息存储到voxel_features数组中
          ((float4 *)voxel_features)[outIndex] = ((float4 *)voxels)[inIndex];
      }

      // clear buffer for next infer
      // 最后,函数使用原子操作将mask数组中该体素的值清零,为下一次处理做准备
      // 清空的目的：下次循环的时候，这里已经没有count了，就不会重复执行特征提取步骤
      atomicExch(mask + voxel_index, 0);
    }

    // create 4 channels
    cudaError_t generateBaseFeatures_launch(unsigned int *mask, float *voxels,
                                            int grid_y_size, int grid_x_size,
                                            unsigned int *pillar_num,
                                            float *voxel_features,
                                            unsigned int *voxel_num,
                                            unsigned int *voxel_idxs,
                                            cudaStream_t stream)
    {
      dim3 threads = {32, 32};
      dim3 blocks = {(grid_x_size + threads.x - 1) / threads.x,
                     (grid_y_size + threads.y - 1) / threads.y};

      generateBaseFeatures_kernel<<<blocks, threads, 0, stream>>>(mask, voxels, grid_y_size, grid_x_size,
                                                                  pillar_num,
                                                                  voxel_features,
                                                                  voxel_num,
                                                                  voxel_idxs);
      cudaError_t err = cudaGetLastError();
      return err;
    }

    // 4 channels -> 10 channels
    static __global__ void generateFeatures_kernel(float *voxel_features,
                                                   unsigned int *voxel_num, unsigned int *voxel_idxs, unsigned int *params,
                                                   float voxel_x, float voxel_y, float voxel_z,
                                                   float range_min_x, float range_min_y, float range_min_z,
                                                   half *features)
    {
      // 这段代码首先计算出当前线程所处理的体素 ID(pillar_idx)和该体素中点的索引(point_idx)。
      // 同时,它还计算出当前线程在块内的索引(pillar_idx_inBlock)。如果当前线程处理的体素 ID 超出了总体素数量,则直接返回
      int pillar_idx = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / WARP_SIZE;
      int point_idx = threadIdx.x % WARP_SIZE;

      int pillar_idx_inBlock = threadIdx.x / WARP_SIZE;
      unsigned int num_pillars = params[0];

      if (pillar_idx >= num_pillars)
        return;

      __shared__ float4 pillarSM[WARPS_PER_BLOCK][WARP_SIZE]; // feats*poitnsNum = 4*32
      __shared__ float4 pillarSumSM[WARPS_PER_BLOCK];
      __shared__ uint4 idxsSM[WARPS_PER_BLOCK];
      __shared__ int pointsNumSM[WARPS_PER_BLOCK];
      __shared__ half pillarOutSM[WARPS_PER_BLOCK][WARP_SIZE][FEATURES_SIZE];

      if (threadIdx.x < WARPS_PER_BLOCK)
      {
        pointsNumSM[threadIdx.x] = voxel_num[blockIdx.x * WARPS_PER_BLOCK + threadIdx.x];
        idxsSM[threadIdx.x] = ((uint4 *)voxel_idxs)[blockIdx.x * WARPS_PER_BLOCK + threadIdx.x];
        pillarSumSM[threadIdx.x] = {0, 0, 0, 0};
      }

      pillarSM[pillar_idx_inBlock][point_idx] = ((float4 *)voxel_features)[pillar_idx * WARP_SIZE + point_idx];

      /*
        在这段代码中,__syncthreads() 的作用是确保所有线程在执行下一步操作之前都已经完成了对 pillarSumSM 的更新操作。

        具体来说:

        每个线程根据自己的 point_idx 和 pillar_idx_inBlock 来判断是否需要更新 pillarSumSM 中对应的值。

        如果需要更新,线程会使用 atomicAdd 函数来原子性地更新 pillarSumSM 中的 x、y、z 三个分量。

        在所有需要更新的线程完成更新操作之后,__syncthreads() 会确保所有线程都已经到达这个同步点,然后才允许线程继续执行后续的代码。

        这样做的目的是为了确保在计算下一步操作之前,pillarSumSM 中的值已经被正确地更新了。如果没有 __syncthreads() 同步,可能会出现数据竞争的问题,导致最终结果不正确。

        总之,这里使用 __syncthreads() 是为了保证并行执行的正确性,是 CUDA 编程中常见的同步手段。
      */
      __syncthreads();

      // calculate sm in a pillar
      if (point_idx < pointsNumSM[pillar_idx_inBlock])
      {
        atomicAdd(&(pillarSumSM[pillar_idx_inBlock].x), pillarSM[pillar_idx_inBlock][point_idx].x);
        atomicAdd(&(pillarSumSM[pillar_idx_inBlock].y), pillarSM[pillar_idx_inBlock][point_idx].y);
        atomicAdd(&(pillarSumSM[pillar_idx_inBlock].z), pillarSM[pillar_idx_inBlock][point_idx].z);
      }
      __syncthreads();

      // 这段代码首先计算出每个点相对于体素中心的偏移量,存储在 mean 变量中。
      // 然后,它计算出每个体素的中心坐标,并将每个点的坐标减去中心坐标,得到相对于体素中心的偏移量,存储在 center 变量中。
      // feature-mean
      float4 mean;
      float validPoints = pointsNumSM[pillar_idx_inBlock];
      mean.x = pillarSumSM[pillar_idx_inBlock].x / validPoints;
      mean.y = pillarSumSM[pillar_idx_inBlock].y / validPoints;
      mean.z = pillarSumSM[pillar_idx_inBlock].z / validPoints;

      mean.x = pillarSM[pillar_idx_inBlock][point_idx].x - mean.x;
      mean.y = pillarSM[pillar_idx_inBlock][point_idx].y - mean.y;
      mean.z = pillarSM[pillar_idx_inBlock][point_idx].z - mean.z;

      // calculate offset
      float x_offset = voxel_x / 2 + idxsSM[pillar_idx_inBlock].w * voxel_x + range_min_x;
      float y_offset = voxel_y / 2 + idxsSM[pillar_idx_inBlock].z * voxel_y + range_min_y;
      float z_offset = voxel_z / 2 + idxsSM[pillar_idx_inBlock].y * voxel_z + range_min_z;

      // feature-offset
      float4 center;
      center.x = pillarSM[pillar_idx_inBlock][point_idx].x - x_offset;
      center.y = pillarSM[pillar_idx_inBlock][point_idx].y - y_offset;
      center.z = pillarSM[pillar_idx_inBlock][point_idx].z - z_offset;

      // store output
      if (point_idx < pointsNumSM[pillar_idx_inBlock])
      {
        pillarOutSM[pillar_idx_inBlock][point_idx][0] = __float2half(pillarSM[pillar_idx_inBlock][point_idx].x);
        pillarOutSM[pillar_idx_inBlock][point_idx][1] = __float2half(pillarSM[pillar_idx_inBlock][point_idx].y);
        pillarOutSM[pillar_idx_inBlock][point_idx][2] = __float2half(pillarSM[pillar_idx_inBlock][point_idx].z);
        pillarOutSM[pillar_idx_inBlock][point_idx][3] = __float2half(pillarSM[pillar_idx_inBlock][point_idx].w);

        pillarOutSM[pillar_idx_inBlock][point_idx][4] = __float2half(mean.x);
        pillarOutSM[pillar_idx_inBlock][point_idx][5] = __float2half(mean.y);
        pillarOutSM[pillar_idx_inBlock][point_idx][6] = __float2half(mean.z);

        pillarOutSM[pillar_idx_inBlock][point_idx][7] = __float2half(center.x);
        pillarOutSM[pillar_idx_inBlock][point_idx][8] = __float2half(center.y);
        pillarOutSM[pillar_idx_inBlock][point_idx][9] = __float2half(center.z);
      }
      else
      {
        pillarOutSM[pillar_idx_inBlock][point_idx][0] = 0;
        pillarOutSM[pillar_idx_inBlock][point_idx][1] = 0;
        pillarOutSM[pillar_idx_inBlock][point_idx][2] = 0;
        pillarOutSM[pillar_idx_inBlock][point_idx][3] = 0;

        pillarOutSM[pillar_idx_inBlock][point_idx][4] = 0;
        pillarOutSM[pillar_idx_inBlock][point_idx][5] = 0;
        pillarOutSM[pillar_idx_inBlock][point_idx][6] = 0;

        pillarOutSM[pillar_idx_inBlock][point_idx][7] = 0;
        pillarOutSM[pillar_idx_inBlock][point_idx][8] = 0;
        pillarOutSM[pillar_idx_inBlock][point_idx][9] = 0;
      }

      __syncthreads();

      for (int i = 0; i < FEATURES_SIZE; i++)
      {
        int outputSMId = pillar_idx_inBlock * WARP_SIZE * FEATURES_SIZE + i * WARP_SIZE + point_idx;
        int outputId = pillar_idx * WARP_SIZE * FEATURES_SIZE + i * WARP_SIZE + point_idx;
        features[outputId] = ((half *)pillarOutSM)[outputSMId];
      }
    }

    nvtype::Int3 VoxelizationParameter::compute_grid_size(const nvtype::Float3 &max_range, const nvtype::Float3 &min_range,
                                                          const nvtype::Float3 &voxel_size)
    {
      nvtype::Int3 size;
      size.x = static_cast<int>(std::round((max_range.x - min_range.x) / voxel_size.x));
      size.y = static_cast<int>(std::round((max_range.y - min_range.y) / voxel_size.y));
      size.z = static_cast<int>(std::round((max_range.z - min_range.z) / voxel_size.z));
      return size;
    }

    cudaError_t generateFeatures_launch(float *voxel_features,
                                        unsigned int *voxel_num,
                                        unsigned int *voxel_idxs,
                                        unsigned int *params, unsigned int max_voxels,
                                        float voxel_x, float voxel_y, float voxel_z,
                                        float range_min_x, float range_min_y, float range_min_z,
                                        nvtype::half *features,
                                        cudaStream_t stream)
    {
      dim3 blocks((max_voxels + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
      dim3 threads(WARPS_PER_BLOCK * WARP_SIZE);

      generateFeatures_kernel<<<blocks, threads, 0, stream>>>(voxel_features,
                                                              voxel_num,
                                                              voxel_idxs,
                                                              params,
                                                              voxel_x, voxel_y, voxel_z,
                                                              range_min_x, range_min_y, range_min_z,
                                                              (half *)features);

      cudaError_t err = cudaGetLastError();
      return err;
    }

    class VoxelizationImplement : public Voxelization
    {
    public:
      virtual ~VoxelizationImplement()
      {
        if (voxel_features_)
          checkRuntime(cudaFree(voxel_features_));
        if (voxel_num_)
          checkRuntime(cudaFree(voxel_num_));
        if (voxel_idxs_)
          checkRuntime(cudaFree(voxel_idxs_));

        if (features_input_)
          checkRuntime(cudaFree(features_input_));
        if (params_input_)
          checkRuntime(cudaFree(params_input_));

        if (mask_)
          checkRuntime(cudaFree(mask_));
        if (voxels_)
          checkRuntime(cudaFree(voxels_));
        if (voxelsList_)
          checkRuntime(cudaFree(voxelsList_));
      }

      bool init(VoxelizationParameter param)
      {
        param_ = param;

        mask_size_ = param_.grid_size.z * param_.grid_size.y * param_.grid_size.x * sizeof(unsigned int);
        voxels_size_ = param_.grid_size.z * param_.grid_size.y * param_.grid_size.x * param_.max_points_per_voxel * 4 * sizeof(float);
        voxel_features_size_ = param_.max_voxels * param_.max_points_per_voxel * 4 * sizeof(float);
        voxel_num_size_ = param_.max_voxels * sizeof(unsigned int);
        voxel_idxs_size_ = param_.max_voxels * 4 * sizeof(unsigned int);
        features_input_size_ = param_.max_voxels * param_.max_points_per_voxel * 10 * sizeof(nvtype::half);

        checkRuntime(cudaMalloc((void **)&voxel_features_, voxel_features_size_));
        checkRuntime(cudaMalloc((void **)&voxel_num_, voxel_num_size_));

        checkRuntime(cudaMalloc((void **)&features_input_, features_input_size_));
        checkRuntime(cudaMalloc((void **)&voxel_idxs_, voxel_idxs_size_));
        checkRuntime(cudaMalloc((void **)&params_input_, sizeof(unsigned int)));

        checkRuntime(cudaMalloc((void **)&mask_, mask_size_));
        checkRuntime(cudaMalloc((void **)&voxels_, voxels_size_));
        checkRuntime(cudaMalloc((void **)&voxelsList_, param_.max_points * sizeof(int)));

        checkRuntime(cudaMemset(voxel_features_, 0, voxel_features_size_));
        checkRuntime(cudaMemset(voxel_num_, 0, voxel_num_size_));

        checkRuntime(cudaMemset(mask_, 0, mask_size_));
        checkRuntime(cudaMemset(voxels_, 0, voxels_size_));
        checkRuntime(cudaMemset(voxelsList_, 0, param_.max_points * sizeof(int)));

        checkRuntime(cudaMemset(features_input_, 0, features_input_size_));
        checkRuntime(cudaMemset(voxel_idxs_, 0, voxel_idxs_size_));

        return true;
      }

      // points and voxels must be of half type
      virtual void forward(const float *_points, int num_points, void *stream) override
      {
        cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

        checkRuntime(cudaMemsetAsync(params_input_, 0, sizeof(unsigned int), _stream));

        // 随机体素生成
        checkRuntime(generateVoxels_random_launch(_points, num_points,
                                                  param_.min_range.x, param_.max_range.x,
                                                  param_.min_range.y, param_.max_range.y,
                                                  param_.min_range.z, param_.max_range.z,
                                                  param_.voxel_size.x, param_.voxel_size.y, param_.voxel_size.z,
                                                  param_.grid_size.y, param_.grid_size.x,
                                                  mask_, voxels_, _stream));
        // voxel_features_: 4 channel
        checkRuntime(generateBaseFeatures_launch(mask_, voxels_,
                                                 param_.grid_size.y, param_.grid_size.x,
                                                 params_input_,
                                                 voxel_features_,
                                                 voxel_num_,
                                                 voxel_idxs_, _stream));
        // voxel_features_: 10 channel
        checkRuntime(generateFeatures_launch(voxel_features_,
                                             voxel_num_,
                                             voxel_idxs_,
                                             params_input_, param_.max_voxels,
                                             param_.voxel_size.x, param_.voxel_size.y, param_.voxel_size.z,
                                             param_.min_range.x, param_.min_range.y, param_.min_range.z,
                                             features_input_, _stream));
      }

      virtual const nvtype::half *features() override { return features_input_; }

      virtual const unsigned int *coords() override { return voxel_idxs_; }

      virtual const unsigned int *params() override { return params_input_; }

    private:
      VoxelizationParameter param_;

      unsigned int *mask_ = nullptr;
      float *voxels_ = nullptr;
      int *voxelsList_ = nullptr;
      float *voxel_features_ = nullptr;
      unsigned int *voxel_num_ = nullptr;

      nvtype::half *features_input_ = nullptr;
      unsigned int *voxel_idxs_ = nullptr; // 记录当前Voxel的坐标信息，后续scatter的时候需要根据坐标信息将特征放到稀疏矩阵中去
      unsigned int *params_input_ = nullptr;

      unsigned int mask_size_;
      unsigned int voxels_size_;
      unsigned int voxel_features_size_;
      unsigned int voxel_num_size_;
      unsigned int voxel_idxs_size_;
      unsigned int features_input_size_ = 0;
    };

    std::shared_ptr<Voxelization> create_voxelization(VoxelizationParameter param)
    {
      std::shared_ptr<VoxelizationImplement> impl(new VoxelizationImplement());
      if (!impl->init(param))
      {
        impl.reset();
      }
      return impl;
    }

  }; // namespace lidar
};   // namespace pointpillar
