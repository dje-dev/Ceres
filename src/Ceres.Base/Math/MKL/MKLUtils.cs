#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directive

using System;

#endregion

namespace Ceres.Base.Math.MKL
{
  /// <summary>
  /// Static helper methods related to Intel MKL library.
  /// </summary>
  public static unsafe class MKLUtils
  {
    /// <summary>
    /// Matrix multiplication (GEMM).
    /// </summary>
    /// <param name="left"></param>
    /// <param name="right"></param>
    /// <param name="ret"></param>
    /// <param name="overrideNumLeftRows">if not -1, indicates the number of rows in the left array to be used (the remainder are ignored)</param>
    /// <returns></returns>
    public static float[,] FloatMultGEMM(float[,] left, float[,] right, float[,] ret, int overrideNumLeftRows = -1)
    {
      int leftRowDim = overrideNumLeftRows == -1 ? left.GetLength(0) : overrideNumLeftRows;
      int leftColDim = left.GetLength(1);
      int rightColDim = right.GetLength(1);

      MKLMethodsNative.cblas_sgemm2D(MKLMethodsNative.BlasOrderType.Row,
                                      MKLMethodsNative.BlasTransType.NoTrans, MKLMethodsNative.BlasTransType.NoTrans,
                                      leftRowDim, rightColDim, leftColDim, 1, left, leftColDim,
                                      right, rightColDim, 0, ret, rightColDim);
      return ret;
    }


    static int COUNT = 0;


    /// <summary>
    /// Matrix vector multiplication.
    /// </summary>
    /// <param name="left"></param>
    /// <param name="right"></param>
    /// <param name="ret"></param>
    /// <param name="overrideNumLeftRows">if not -1, indicates the number of rows in the left array to be used (the remainder are ignored)</param>
    /// <returns></returns>
    public static float[] FloatMultGEMM1D(float[] left, float[,] right, float[] ret, int overrideNumLeftRows = -1)
    {
      int leftRowDim = 1;
      int leftColDim = left.GetLength(0);
      int rightColDim = right.GetLength(1);

      // Yet to do: try using the version which does not check arguments for speed
      MKLMethodsNative.cblas_sgemm1D2D(MKLMethodsNative.BlasOrderType.Row,
                                      MKLMethodsNative.BlasTransType.NoTrans, MKLMethodsNative.BlasTransType.NoTrans,
                                      leftRowDim, rightColDim, leftColDim, 1, left, leftColDim,
                                      right, rightColDim, 0, ret, rightColDim);
      return ret;
    }


    public static void FloatMultGEMM(float* left, float* right, int leftRowDim, int leftColDim, int rightRowDim, int rightColDim, float* ret)
    {
      if (leftColDim != rightRowDim) throw new Exception("wrong dim");

      MKLMethodsNative.cblas_sgemm_raw(MKLMethodsNative.BlasOrderType.Row, MKLMethodsNative.BlasTransType.NoTrans, MKLMethodsNative.BlasTransType.NoTrans,
                                      leftRowDim, rightColDim, leftColDim, 1, (IntPtr)left, leftColDim,
                                      (IntPtr)right, rightColDim, 0, (IntPtr)ret, rightColDim);
    }


    [ThreadStatic] static float[] bufferValues = null;
    [ThreadStatic] static int[] bufferCols = null;
    [ThreadStatic] static int[] bufferRows = null;


    /// <summary>
    /// Sparse Matrix/Matrix multiplication.
    /// </summary>
    /// <param name="left"></param>
    /// <param name="right"></param>
    /// <param name="ret"></param>
    /// <param name="maxPossibleNonzero"></param>
    public static void SparseMatrixMatrixMult(float[,] left, float[,] right, float[,] ret, int maxPossibleNonzero)
    {
      if (bufferValues == null || bufferValues.Length != maxPossibleNonzero)
      {
        bufferRows = new int[maxPossibleNonzero];
        bufferCols = new int[maxPossibleNonzero];
        bufferValues = new float[maxPossibleNonzero];
      }

      // Use local copies of ThreadStatic variable references for efficiency.
      int[] bufferColsLocal = bufferCols;
      int[] bufferRowsLocal = bufferRows;
      float[] bufferValuesLocal = bufferValues;

      int numRows = left.GetLength(0);
      int numCols = left.GetLength(1);

      // Get nonzero entries.
      // Note that it is not actually necessary to zero out
      // the bufferValues which may not be filled because the MKL
      // routine accepts the numActualNonzero as an argument.
      int numActualNonzero = 0;
      for (int i = 0; i < numRows; i++)
      {
        for (int j = 0; j < numCols; j++)
        {
          if (left[i, j] != 0)
          {
            if (numActualNonzero >= maxPossibleNonzero)
              throw new Exception(maxPossibleNonzero + " maxPossibleNonzero value was insufficient");

            bufferRowsLocal[numActualNonzero] = i;
            bufferColsLocal[numActualNonzero] = j;
            bufferValuesLocal[numActualNonzero] = left[i, j];
            numActualNonzero++;
          }
        }
      }

      // "For general matrix you just put 'G' as first element, then 2nd, and 3rd are ignored anyway, so put anything, 
      // and 'F'(for 1 - based indexing) or 'C'(for zero - based) as the last one"
      char[] flg = new char[] { 'g', 'x', 'x', 'c' };
      char fft = 'n';

      int m = left.GetLength(0);
      int n = right.GetLength(1);
      int k = left.GetLength(1);
      float alpha = 1.0f;
      float beta = 0.0f;
      MKLMethodsNative.cblas_scoomm(ref fft,
                              ref m, ref n, ref k,
                              ref alpha,
                              flg, bufferValues, bufferRows, bufferCols, ref numActualNonzero,
                              right, ref n, ref beta, ret, ref n);
    }


    /// <summary>
    /// Performs spectral (Eignevector) decomposition.
    /// </summary>
    /// <param name="m"></param>
    /// <param name="eigenvalues"></param>
    /// <param name="calcEigenvectors"></param>
    /// <param name="maxEigenvalues"></param>
    /// <returns></returns>
    public static float[,] Eigendecompose(float[,] m, ref float[] eigenvalues, bool calcEigenvectors, int maxEigenvalues = 0)
    {
      int NUM_ROWS = m.GetLength(0);
      int NUM_COLS = m.GetLength(1);

      char type = calcEigenvectors ? 'V' : 'N';
      char range = 'I';
      char target = 'U';
      int lda = NUM_ROWS;

      // Define return values
      eigenvalues = new float[lda];
      float[,] eigenvectors = calcEigenvectors ? new float[lda, lda] : null;

      int eigenCount = 0;
      float abstol = 0.0f;
      int scratchLen = -1;
      float[] scratch = new float[lda];
      int scratchLenI = -1;
      int[] scratchI = new int[lda];
      float[] scratch1 = new float[1];
      float[] scratch2 = new float[1];
      int[] scratch1i = new int[1];
      int[] scratch2i = new int[1];
      int[] support = new int[lda * 2];

      scratch1i[0] = NUM_ROWS - maxEigenvalues + 1;
      scratch2i[0] = NUM_ROWS;

      float[,] dataTranspose = Copy(m, true);

      // Scratch area
      int errCodeScratch = 0;
      MKLMethodsNative.SSYEVR(ref type, ref range, ref target, ref lda, dataTranspose, ref lda,
                                 scratch1, scratch2, scratch1i, scratch2i,
                                 ref abstol, ref eigenCount, eigenvalues, eigenvectors, ref lda, support,
                                 scratch, ref scratchLen, scratchI, ref scratchLenI, ref errCodeScratch);
      if (errCodeScratch != 0) throw new Exception("SSYEVR MKL Failure:  " + errCodeScratch);

      scratchLen = (int)scratch[0];
      scratch = new float[scratchLen];
      scratchLenI = (int)scratchI[0];
      scratchI = new int[scratchLenI];

      int errCode = 0;
      MKLMethodsNative.SSYEVR(ref type, ref range, ref target, ref lda, dataTranspose, ref lda,
                                 scratch1, scratch2, scratch1i, scratch2i,
                                 ref abstol, ref eigenCount, eigenvalues, eigenvectors, ref lda, support,
                                 scratch, ref scratchLen, scratchI, ref scratchLenI, ref errCode);
      if (errCode != 0) throw new Exception("SSYEVR2D  MKL Failure: " + errCode);

      Array.Reverse(eigenvalues);
      return eigenvectors;
    }


    public static float[,] Copy(float[,] input, bool transpose)
    {
      int rows = input.GetLength(0);
      int cols = input.GetLength(1);
      float[,] ret = transpose ? new float[cols, rows] : new float[rows, cols];
      for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
          if (!transpose)
            ret[i, j] = input[i, j];
          else
            ret[j, i] = input[i, j];

      return ret;
    }

  }
}
