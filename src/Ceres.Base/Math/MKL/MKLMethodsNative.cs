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
using System.Runtime.InteropServices;
using System.Security;

#endregion

namespace Ceres.Base.Math.MKL
{
  /// <summary>
  /// Static interopability function for Intel MKL.
  /// </summary>
  [SuppressUnmanagedCodeSecurity]
  public static unsafe class MKLMethodsNative
  {
    /// <summary>
    /// Matrix transpose type.
    /// </summary>
    public enum BlasTransType { NoTrans = 111, Trans = 112, Packed = 151/* ? */ }
    public enum BlasOrderType { Row = 101, Column = 102 }
    public enum BlasMatrixID { A = 161, B = 162, C = 163 }
    public enum BlasOffset { Row = 171, Col = 172, Fixed = 173 }

    const bool MKL_VERBOSE = false;


    // TODO: generalize this.
    private const string dllName = @"C:\mkl\works\mkl_rt.dll";


    static MKLMethodsNative()
    {
      // NOTE! Perforamnce is dramatically degraded without these settings.
      System.Environment.SetEnvironmentVariable("KMP_AFFINITY", "physical");
      System.Environment.SetEnvironmentVariable("MKL_NUM_THREADS", "1");

      // Use this to turn on VERBOSE  mode, was not able to get the mkl_verbose API call to work.
      if (MKL_VERBOSE) System.Environment.SetEnvironmentVariable("MKL_VERBOSE", "1");
    }

    [DllImport(dllName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
    public static extern int vdExp(int N, [In] double[] A, [Out] double[] Y);

    [DllImport(dllName, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
    public static extern int vdAdd(int N, [In] double[] A, [In] double[] B, [Out] double[] Y);

    [DllImport(dllName, EntryPoint = "vdAdd", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
    public static extern int vdAddPtr(int N, float* A, float* B, float* C);

    [DllImport(dllName, EntryPoint = "vsAdd", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
    public static extern int vsAddPtr(int N, float* A, float* B, float* C);

    [DllImport(dllName, EntryPoint = "mkl_verbose", ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)] // ** NOT WORKING (?)
    public static extern void mkl_verbose(ref int verbose);

    // PACKED (COULD NOT GET THIS TO WORK)
    [DllImport(dllName, EntryPoint = "cblas_sgemm_alloc", ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr cblas_sgemm_alloc(BlasMatrixID matrixID, int m, int n, int k);

    [DllImport(dllName, EntryPoint = "cblas_sgemm_pack", ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
    public static extern void cblas_sgemm_pack(BlasOrderType order, BlasMatrixID matrixID, BlasTransType transSrc, int m, int n, int k, float alpha, float[,] src, int ld, IntPtr dest);

    [DllImport(dllName, EntryPoint = "cblas_sgemm_pack", ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
    public static extern void cblas_sgemm_pack1(BlasMatrixID matrixID, BlasTransType transSrc, int m, int n, int k, float alpha, float[,] src, int ld, IntPtr dest);

    [DllImport(dllName, EntryPoint = "cblas_sgemm_compute", ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
    public static extern void cblas_compute(BlasOrderType order, BlasTransType transA, BlasTransType transB, int m, int n, int k, float[,] a,
                                            int lda, /*float[,] b*/ IntPtr bPacked, int ldb, float beta, float[,] C, int ldc);

    [DllImport(dllName, EntryPoint = "cblas_sgemm_free", ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
    public static extern void cblas_sgemm_free(IntPtr dest);
    // END PACKED


    [DllImport(dllName, EntryPoint = "cblas_sgemm", ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
    public static extern void cblas_sgemm2D(BlasOrderType order, BlasTransType transA, BlasTransType transB, int m, int n, int k, float alpha, float[,] a, int lda, float[,] b, int ldb, float beta, [In, Out] float[,] c, int ldc);

    [DllImport(dllName, EntryPoint = "cblas_sgemm", ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
    public static extern void cblas_sgemm1D2D(BlasOrderType order, BlasTransType transA, BlasTransType transB, int m, int n, int k, float alpha, float[] a, int lda, float[,] b, int ldb, float beta, [In, Out] float[] c, int ldc);

    [DllImport(dllName, EntryPoint = "cblas_sgemm", ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
    public static extern void cblas_sgemm1D(BlasOrderType order, BlasTransType transA, BlasTransType transB, int m, int n, int k, float alpha, float[] a, int lda, float[] b, int ldb, float beta, [In, Out] float[] c, int ldc);

    [DllImport(dllName, EntryPoint = "cblas_sgemm", ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
    public static extern void cblas_sgemm1DPtr(BlasOrderType order, BlasTransType transA, BlasTransType transB, int m, int n, int k, float alpha, float[] a, int lda, float* b, int ldb, float beta, float* c, int ldc);

    [DllImport(dllName, EntryPoint = "cblas_sgemm", ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
    public static extern void cblas_sgemm1DPtrAll(BlasOrderType order, BlasTransType transA, BlasTransType transB, int m, int n, int k, float alpha, float* a, int lda, float* b, int ldb, float beta, float* c, int ldc);


    // this method not in mklmkl, and uses "old style" pointers
    [DllImport(dllName, EntryPoint = "mkl_scoomm", ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
    public static extern void cblas_scoomm(ref char transT, ref int m, ref int n, ref int k, ref float alpha,
                                       char[] matdescra,
                                       float[] values, int[] rowInd, int[] colInd, ref int numNonzero,
                                       float[,] b, ref int ldb, ref float beta, [In, Out] float[,] c, ref int ldc);

    [DllImport(dllName, EntryPoint = "mkl_scoomv", ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
    public static extern void cblas_scoomv(ref char transT, ref int m, ref int k, ref float alpha,
                                       char[] matdescra,
                                       float[] values, int[] rowInd, int[] colInd, ref int numNonzero,
                                       float[,] b, ref float beta, [In, Out] float[] c);


    [DllImport(dllName, EntryPoint = "cblas_sgemm", ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
    public static extern void cblas_sgemm_raw(BlasOrderType order, BlasTransType transA, BlasTransType transB, int m, int n, int k, float alpha, IntPtr a, int lda, IntPtr b, int ldb, float beta, [In, Out] IntPtr c, int ldc);


    [DllImport(dllName, EntryPoint = "cblas_sgemv", ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
    public static extern void cblas_sgemv2D(BlasOrderType order, BlasTransType transA, int m, int n, float alpha, float[,] a, int lda, float[] x, int incX, float beta, [In, Out] float[] y, int incY);

    [DllImport(dllName, ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
    public static extern void cblas_sgemv(BlasOrderType order, BlasTransType transA, int m, int n, float alpha, float[] a, int lda, float[] x, int incX, float beta, [In, Out] float[] y, int incY);


    [DllImport(dllName, EntryPoint = "SSYEVR", ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
    public static extern void SSYEVR(ref char jobz, ref char range, ref char uplo, ref int n, [In] float[,] a, ref int lda,
                                 [In] float[] vl, [In] float[] vu, [In] int[] il, [In] int[] iu, ref float abstol, ref int m, [In, Out] float[] w, [In, Out] float[,] z, ref int ldz,
                                 [In, Out] int[] isuppz, [In, Out] float[] work, ref int len, [In, Out] int[] iwork, ref int ilen, ref int info);


    [DllImport(dllName, EntryPoint = "cblas_gemm_s16s16s32", ExactSpelling = true, SetLastError = false, CallingConvention = CallingConvention.Cdecl)]
    public static extern void cblas_gemm_s16s16s32(BlasOrderType order, BlasTransType transA, BlasTransType transB, MKLMethodsNative.BlasOffset offset,
                                                   int m, int n, int k,
                                                   float alpha, short[,] a, int lda, short ao,
                                                   short[,] b, int ldb, short bo,
                                                   float beta, int[] c, int ldc, int[] co);
  }
}
