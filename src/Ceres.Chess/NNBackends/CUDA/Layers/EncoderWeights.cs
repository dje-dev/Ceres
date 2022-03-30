#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directives

using ManagedCuda;
using Ceres.Base.DataTypes;
using System.Runtime.InteropServices;
using System;

#endregion

// NOTE: This file is a highly derivative of the LC0 CUDA backend source code in the Leela Chess Zero project
//       constituting largely a transliteration of C++ code into C#, with certain enhancements.

namespace Ceres.Chess.NNBackends.CUDA
{
  public class EncoderWeights : IDisposable
  {
    public CudaDeviceVariable<FP16> mha_q_w;
    public CudaDeviceVariable<FP16> mha_q_b;
    public CudaDeviceVariable<FP16> mha_k_w;
    public CudaDeviceVariable<FP16> mha_k_b;
    public CudaDeviceVariable<FP16> mha_v_w;
    public CudaDeviceVariable<FP16> mha_v_b;
    public CudaDeviceVariable<FP16> mha_dense_w;
    public CudaDeviceVariable<FP16> mha_dense_b;

    public CudaDeviceVariable<FP16> mha_qkv_w;
    public CudaDeviceVariable<FP16> mha_qkv_b;

    public CudaDeviceVariable<FP16> ln1_betas;
    public CudaDeviceVariable<FP16> ln1_gammas;
    public CudaDeviceVariable<FP16> ln2_betas;
    public CudaDeviceVariable<FP16> ln2_gammas;

    public CudaDeviceVariable<FP16> ffn_dense1_w;
    public CudaDeviceVariable<FP16> ffn_dense1_b;
    public CudaDeviceVariable<FP16> ffn_dense2_w;
    public CudaDeviceVariable<FP16> ffn_dense2_b;


    public int mha_q_size;
    public int mha_k_size;
    public int mha_v_size;
    public int mha_dense_size;
    public int ffn_dense1_size;
    public int ffn_dense2_size;

  public EncoderWeights(in LC0LegacyWeights.EncoderWeights weights)
    {
      mha_q_size = weights.mha.q_b.Length;
      mha_k_size = weights.mha.k_b.Length;
      mha_v_size = weights.mha.v_b.Length;
      mha_dense_size = weights.mha.dense_b.Length;
      ffn_dense1_size = weights.ffn.dense1_b.Length;
      ffn_dense2_size = weights.ffn.dense2_b.Length;

      mha_q_w = BaseLayerCUDA.LoadedWeights(weights.mha.q_w);
      mha_q_b = BaseLayerCUDA.LoadedWeights(weights.mha.q_b);
      mha_k_w = BaseLayerCUDA.LoadedWeights(weights.mha.k_w);
      mha_k_b = BaseLayerCUDA.LoadedWeights(weights.mha.k_b);
      mha_v_w = BaseLayerCUDA.LoadedWeights(weights.mha.v_w);
      mha_v_b = BaseLayerCUDA.LoadedWeights(weights.mha.v_b);
      mha_dense_w = BaseLayerCUDA.LoadedWeights(weights.mha.dense_w);
      mha_dense_b = BaseLayerCUDA.LoadedWeights(weights.mha.dense_b);

      // big allocation to hold qkv weights one after the other
      int elements = weights.mha.q_w.Length;
      int size = elements * Marshal.SizeOf<FP16>() * 3;
      mha_qkv_w = new CudaDeviceVariable<FP16>(elements * 3);
      mha_qkv_w.CopyToDevice(mha_q_w, 0, 0,          size/3);
      mha_qkv_w.CopyToDevice(mha_k_w, 0, 1 * size/3, size/3);
      mha_qkv_w.CopyToDevice(mha_v_w, 0, 2 * size/3, size/3);

      elements = weights.mha.q_b.Length;
      size = elements * Marshal.SizeOf<FP16>() * 3;
      mha_qkv_b = new CudaDeviceVariable<FP16>(elements * 3);
      mha_qkv_b.CopyToDevice(mha_q_b, 0, 0, size / 3);
      mha_qkv_b.CopyToDevice(mha_k_b, 0, 1 * size / 3, size / 3);
      mha_qkv_b.CopyToDevice(mha_v_b, 0, 2 * size / 3, size / 3);

      ln1_betas = BaseLayerCUDA.LoadedWeights(weights.ln1_betas);
      ln1_gammas = BaseLayerCUDA.LoadedWeights(weights.ln1_gammas);
      ln2_betas = BaseLayerCUDA.LoadedWeights(weights.ln2_betas);
      ln2_gammas = BaseLayerCUDA.LoadedWeights(weights.ln2_gammas);

      ffn_dense1_w = BaseLayerCUDA.LoadedWeights(weights.ffn.dense1_w);
      ffn_dense1_b = BaseLayerCUDA.LoadedWeights(weights.ffn.dense1_b);
      ffn_dense2_w = BaseLayerCUDA.LoadedWeights(weights.ffn.dense2_w);
      ffn_dense2_b = BaseLayerCUDA.LoadedWeights(weights.ffn.dense2_b);
    }

    public void Dispose()
    {
      mha_q_w?.Dispose();
      mha_q_b.Dispose();
      mha_k_w.Dispose();
      mha_k_b.Dispose();
      mha_v_w.Dispose();
      mha_v_b.Dispose();
      mha_dense_w.Dispose();
      mha_dense_b.Dispose();

      mha_qkv_w.Dispose();
      mha_qkv_b.Dispose();

      ln1_betas.Dispose();
      ln1_gammas.Dispose();
      ln2_betas.Dispose();
      ln2_gammas.Dispose();

      ffn_dense1_w.Dispose();
      ffn_dense1_b.Dispose();
      ffn_dense2_w.Dispose();
      ffn_dense2_b.Dispose();
    }

  }
}
