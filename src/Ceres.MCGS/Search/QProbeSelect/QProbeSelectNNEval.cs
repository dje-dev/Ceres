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
using System.Diagnostics;
using System.Numerics.Tensors;

#endregion

namespace Ceres.MCGS.Search.QProbeSelect;

/// <summary>
/// Per-thread scratch buffers for QProbeSelectNNEval. Small (a few KB per thread);
/// sized on first use for the loaded model's geometry.
/// </summary>
public sealed class QProbeSelectScratch
{
  internal float[] Inputs;
  internal float[] BufA;
  internal float[] BufB;

  internal void EnsureSized(QProbeSelectNNModel model)
  {
    if (Inputs == null || Inputs.Length < model.NumInputs)
    {
      Inputs = new float[model.NumInputs];
    }
    if (BufA == null || BufA.Length < model.HiddenDim)
    {
      BufA = new float[model.HiddenDim];
      BufB = new float[model.HiddenDim];
    }
  }
}


/// <summary>
/// Dependency-free inline forward pass for the Q-uncertainty MLP (.qnn v3), mirroring
/// the NumPy reference evaluator evaluate_qnn in qprobetrainpy/exportinline.py exactly
/// (which is itself parity-gated against the PyTorch checkpoint at export time).
/// All arithmetic fp32; SIMD via System.Numerics.Tensors.TensorPrimitives.
/// The model is immutable; all mutable state lives in the caller-supplied scratch,
/// so concurrent threads may evaluate concurrently with per-thread scratch.
/// </summary>
public static class QProbeSelectNNEval
{
  /// <summary>
  /// Evaluates the model on the RAW input row in scratch.Inputs (length model.NumInputs:
  /// capture features followed by the 2 dN-conditioning inputs log1p(dN) and dN/64).
  /// Returns (mu, sigma) in dQ units; sigma is defensively clamped.
  /// </summary>
  public static (float Mu, float Sigma) Evaluate(QProbeSelectNNModel model, QProbeSelectScratch scratch)
  {
    scratch.EnsureSized(model);
    bool relu = model.Activation == QProbeSelectNNModel.ActivationKind.ReLU;
    float eps = model.RMSEps;
    int hidden = model.HiddenDim;

    ReadOnlySpan<float> x = scratch.Inputs.AsSpan(0, model.NumInputs);
    Span<float> a = scratch.BufA.AsSpan(0, hidden);
    Span<float> b = scratch.BufB.AsSpan(0, hidden);

    for (int layer = 0; layer < model.NumHiddenLayers; layer++)
    {
      ReadOnlySpan<float> src = layer == 0 ? x : (layer % 2 == 1 ? a : b);
      Span<float> dst = layer % 2 == 0 ? a : b;
      Dense(src, model.MlpW[layer], model.MlpB[layer], dst, hidden);
      RMSNorm(dst, model.MlpRms[layer], eps);
      Activate(dst, relu);
    }

    ReadOnlySpan<float> final = (model.NumHiddenLayers % 2 == 1) ? a : b;
    float mu = TensorPrimitives.Dot(final, model.HeadW.AsSpan(0, hidden)) + model.HeadB[0];
    float logSigma = TensorPrimitives.Dot(final, model.HeadW.AsSpan(hidden, hidden)) + model.HeadB[1];
    float sigma = Math.Clamp(MathF.Exp(logSigma),
                             QProbeSelectNNModel.SIGMA_MIN, QProbeSelectNNModel.SIGMA_MAX);
    return (mu, sigma);
  }


  /// <summary>
  /// Fills the 2 dN-conditioning inputs (appended after the capture features) for a
  /// query horizon dN. Mirrors data.dn_inputs in the Python pipeline.
  /// </summary>
  public static void SetDnInputs(QProbeSelectNNModel model, QProbeSelectScratch scratch, int dn)
  {
    Debug.Assert(dn >= 1);
    scratch.Inputs[model.NumInputs - 2] = MathF.Log(1f + dn);
    scratch.Inputs[model.NumInputs - 1] = dn / 64f;
  }


  /// <summary>
  /// Batched evaluation over numRows RAW input rows stored row-major in
  /// inputsRowMajor (row length model.NumInputs). Intended usage at select time:
  /// build one row per (child, query dN) for ALL children of a parent and evaluate
  /// them in a single call, so the (L2-resident) weight matrices are streamed once
  /// per batch rather than reloaded per child, and downstream code gets flat
  /// mu[]/sigma[] arrays to score against. Writes outputs in row order.
  /// </summary>
  public static void EvaluateBatch(QProbeSelectNNModel model, QProbeSelectScratch scratch,
                                   ReadOnlySpan<float> inputsRowMajor, int numRows,
                                   Span<float> mu, Span<float> sigma)
  {
    Debug.Assert(inputsRowMajor.Length >= numRows * model.NumInputs);
    Debug.Assert(mu.Length >= numRows && sigma.Length >= numRows);
    scratch.EnsureSized(model);

    for (int row = 0; row < numRows; row++)
    {
      inputsRowMajor.Slice(row * model.NumInputs, model.NumInputs)
                    .CopyTo(scratch.Inputs.AsSpan(0, model.NumInputs));
      (mu[row], sigma[row]) = Evaluate(model, scratch);
    }
  }


  /// <summary>
  /// Dense matvec y = W x + b with W stored [nout, nin] row-major
  /// (each output neuron's weights are one contiguous span).
  /// </summary>
  static void Dense(ReadOnlySpan<float> x, float[] weights, float[] bias, Span<float> y, int nout)
  {
    int nin = x.Length;
    for (int o = 0; o < nout; o++)
    {
      y[o] = TensorPrimitives.Dot(x, weights.AsSpan(o * nin, nin)) + bias[o];
    }
  }


  /// <summary>
  /// In-place RMSNorm: row = weight * row / sqrt(mean(row^2) + eps).
  /// </summary>
  static void RMSNorm(Span<float> row, float[] weight, float eps)
  {
    float meanSquare = TensorPrimitives.Dot(row, row) / row.Length;
    float inv = 1f / MathF.Sqrt(meanSquare + eps);
    TensorPrimitives.Multiply(row, inv, row);
    TensorPrimitives.Multiply(row, weight.AsSpan(0, row.Length), row);
  }


  /// <summary>
  /// In-place activation: ReLU (default) or SiLU per the model header.
  /// </summary>
  static void Activate(Span<float> row, bool relu)
  {
    if (relu)
    {
      TensorPrimitives.Max(row, 0f, row);
    }
    else
    {
      for (int i = 0; i < row.Length; i++)
      {
        row[i] = row[i] / (1f + MathF.Exp(-row[i]));
      }
    }
  }
}
