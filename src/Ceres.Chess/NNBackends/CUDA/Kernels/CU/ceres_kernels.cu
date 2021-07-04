#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#include "cuda_common.h"
namespace ceres {


__global__ void copyMaskedMovesKernel(half* inPolicies, short* inMovesIndices, float* outPoliciesMasked, int size) 
{
  int itemIndex = threadIdx.x + blockDim.x * blockIdx.x;

  int offsetPolicies = 1858 * itemIndex;
  int offsetMoves = 96 * itemIndex;
  int offsetPoliciesOut = 96 * itemIndex;

  if (itemIndex < size)
  {
    for (int i=0;i<96;i++)
    {
      outPoliciesMasked[offsetPoliciesOut + i] = inPolicies[offsetPolicies + inMovesIndices[offsetMoves + i]];
//      outPoliciesMasked[offsetPoliciesOut + i] = exp((float)inPolicies[offsetPolicies + inMovesIndices[offsetMoves + i]]);
    }
  }

  }


__global__ void shiftConvertKernel(half* outV, char* inData, float minVal, float maxVal, int size) 
{
  int itemIndex = threadIdx.x + blockDim.x * blockIdx.x;
  if (itemIndex < size)
  {
    char b0 = inData[itemIndex * 2];
    char b1 = inData[itemIndex * 2 + 1];
    float v1 = 256 * b1 + b0;

    outV[itemIndex] = (half)(minVal + v1 * (maxVal - minVal) / 65535.0f);
  }
}

}