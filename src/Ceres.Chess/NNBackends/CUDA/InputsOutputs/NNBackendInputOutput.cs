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

using System;
using Ceres.Base.CUDA;
using Ceres.Chess.EncodedPositions;
using ManagedCuda;

#endregion

// NOTE: This file is a highly derivative of the LC0 CUDA backend source code in the Leela Chess Zero project
//       constituting largely a transliteration of C++ code into C#, with certain enhancements.

namespace Ceres.Chess.NNBackends.CUDA
{

  /// <summary>
  /// Set of variables used to exchange data with GPU.
  /// </summary>
  public class NNBackendInputOutput
  {
    public const int NUM_OUTPUT_POLICY = EncodedPolicyVector.POLICY_VECTOR_LENGTH;
    public const int NUM_INPUT_PLANES = 112;

    public const int MAX_MOVES = 96;

    public readonly int MaxBatchSize;

    // Inputs
    internal CUDAPinnedMemory<ulong> InputBoardMasks;
    internal CUDAPinnedMemory<float> InputBoardValues;

    internal short[] InputNumMovesUsed;
    internal CUDAPinnedMemory<short> InputMoveIndices;

    // Outputs
    internal CUDAPinnedMemory<float> OutputPolicyHeadMasked;

    static public float[,] OutputValueHeadRaw; // before exponentiation and normalization. EXPERIMENTAL STATIC

    internal float[,] OutputValueHead;
    internal float[,] OutputValueHeadFC2;

    internal float[] OutputMovesLeftHead;

    // GPU
    internal CudaDeviceVariable<ulong> input_masks_gpu_;
    internal CudaDeviceVariable<float> input_val_gpu_;

    internal NNBackendInputOutput(int maxBatchSize, bool wdl, bool moves_left)
    {
      MaxBatchSize = maxBatchSize;
      
      input_masks_gpu_ = new CudaDeviceVariable<ulong>(maxBatchSize * NUM_INPUT_PLANES * sizeof(ulong));
      input_val_gpu_ = new CudaDeviceVariable<float>(maxBatchSize * NUM_INPUT_PLANES * sizeof(float));

      InputBoardMasks = new CUDAPinnedMemory<ulong>(maxBatchSize * NUM_INPUT_PLANES, true);
      InputBoardValues = new CUDAPinnedMemory<float>(maxBatchSize * NUM_INPUT_PLANES, true);
      InputMoveIndices = new CUDAPinnedMemory<short>(maxBatchSize * MAX_MOVES); // note that not hostToDeviceOnly 
      InputNumMovesUsed = new short[maxBatchSize];

      OutputPolicyHeadMasked = new CUDAPinnedMemory<float>(maxBatchSize * MAX_MOVES);
      OutputValueHead = new float[maxBatchSize, (wdl ? 3 : 1)];

      if (moves_left)
      {
        OutputMovesLeftHead = new float[maxBatchSize];
      }
      else
      {
        OutputMovesLeftHead = null;
      }
    }

  };

}
