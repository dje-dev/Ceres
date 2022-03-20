#region Using directives

using System;
using System.Threading.Tasks;
using Ceres.Chess.LC0.WeightsProtobuf;

#endregion


// NOTE: This file is a highly derivative of the LC0 CUDA backend source code in the Leela Chess Zero project
//       constituting largely a transliteration of C++ code into C#, with certain enhancements.

namespace Ceres.Chess.NNBackends.CUDA
{
  /// <summary>
  /// 
  /// Note that conversion to FP16 is not done here to retain full 
  /// necessary precision in subsequent calculations (batch normalization folding).
  /// </summary>
  public record LC0LegacyWeights
  {
    const float EPSILON = 1e-5f;

    /// <summary>
    /// The largest scale factor used in any GetLayerLiner16 call made.
    /// Extremely large values (such as 500,000) can indicate unstable network
    /// that cannot be processed properly (especially in FP16).
    /// </summary>
    public float LargestScaleSeen;

    float[] WeightsDecoded(Pblczero.Weights.Layer layer)
    {
      if (layer == null)
      {
        return null;

      }

      float scale;
      float[] ret = ProtobufHelpers.GetLayerLinear16(layer, out scale);
      LargestScaleSeen = Math.Max(scale, LargestScaleSeen);
      return ret;
    }

    public LC0LegacyWeights(Pblczero.Weights weights)
    {
      LargestScaleSeen = float.MinValue;

      input = new ConvBlock(this, weights.Input);
      policy1 = weights.Policy1 == null ? default : new ConvBlock(this, weights.Policy1);
      policy = weights.Policy == null ? default : new ConvBlock(this, weights.Policy);
      ip_pol_w = WeightsDecoded(weights.IpPolW);
      ip_pol_b = WeightsDecoded(weights.IpPolB);
      ip2_pol_w = WeightsDecoded(weights.Ip2PolW);
      ip2_pol_b = WeightsDecoded(weights.Ip2PolB);
      ip3_pol_w = WeightsDecoded(weights.Ip3PolW);
      ip3_pol_b = WeightsDecoded(weights.Ip3PolB);
      ip4_pol_w = WeightsDecoded(weights.Ip4PolW);
      numPolicyEncoderHeads = (int)weights.PolHeadcount;


      value = new ConvBlock(this, weights.Value);
      ip1_val_w = WeightsDecoded(weights.Ip1ValW);
      ip1_val_b = WeightsDecoded(weights.Ip1ValB);
      ip2_val_w = WeightsDecoded(weights.Ip2ValW);
      ip2_val_b = WeightsDecoded(weights.Ip2ValB);

#if ACTION_HEAD
      if (weights.Action != null)
      {
        // Experimental action head of jjosh
        action = new ConvBlock(this, weights.Action);
        ip1_action_w = WeightsDecoded(weights.IpActW);
        ip1_action_b = WeightsDecoded(weights.IpActB);
      }
      else
      {
        action = default;
        ip1_action_w = default;
        ip1_action_b = default;
      }
#endif

      moves_left = weights.MovesLeft == null ? default : new ConvBlock(this, weights.MovesLeft);
      ip1_mov_w = WeightsDecoded(weights.Ip1MovW);
      ip1_mov_b = WeightsDecoded(weights.Ip1MovB);
      ip2_mov_w = WeightsDecoded(weights.Ip2MovW);
      ip2_mov_b = WeightsDecoded(weights.Ip2MovB);

      // Decode in parallel for reduced runtime.
      Residual[] tempResiduals = new Residual[weights.Residuals.Count];
      Parallel.For(0, tempResiduals.Length, i => tempResiduals[i] = new Residual(this, weights.Residuals[i]));
      residual = tempResiduals;

      if (weights.PolEncoders != null)
      {
        policyEncoders = new EncoderLayer[weights.PolEncoders.Count];
        Parallel.For(0, policyEncoders.Length, i => policyEncoders[i] = new EncoderLayer(this, weights.PolEncoders[i]));
      }
    }


    public record ConvBlock
    {
      public float[] weights;
      public float[] biases;
      public float[] bn_gammas;
      public float[] bn_betas;
      public float[] bn_means;
      public float[] bn_stddivs;

      public ConvBlock(LC0LegacyWeights parent, Pblczero.Weights.ConvBlock block)
      {
        weights = parent.WeightsDecoded(block.Weights);
        biases = parent.WeightsDecoded(block.Biases);
        bn_gammas = parent.WeightsDecoded(block.BnGammas);
        bn_betas = parent.WeightsDecoded(block.BnBetas);
        bn_means = parent.WeightsDecoded(block.BnMeans);
        bn_stddivs = parent.WeightsDecoded(block.BnStddivs);

        if (weights == null)
        {
          // Empty ConvBlock.
          return;
        }

        if (bn_betas == null)
        {
          bn_betas = new float[biases.Length];
          bn_gammas = new float[biases.Length];

          // Old net without gamma and beta.
          for (int i = 0; i < biases.Length; i++) 
          {
            bn_betas[i] = 0.0f;
            bn_gammas[i] = 1.0f;
          }
        }


        if (biases == null)
        {
          biases = new float[bn_means.Length];
        }

        if (bn_means == null)
        {
          // No batch norm.
          return;
        }

        // Fold batch norm into weights and biases.
        // Variance to gamma.
        for (int i = 0; i < bn_stddivs.Length; i++) 
        {
          bn_gammas[i] *= 1.0f / MathF.Sqrt(bn_stddivs[i] + EPSILON);
          bn_means[i] -= biases[i];
        }

        int outputs = biases.Length;

        // We can treat the [inputs, filter_size, filter_size] dimensions as one.
        int inputs = weights.Length / outputs;

        for (int o = 0; o < outputs; o++) 
        {  
          for (int c = 0; c < inputs; c++) 
          {
            weights[o * inputs + c] *= bn_gammas[o];
          }

          biases[o] = -bn_gammas[o] * bn_means[o] + bn_betas[o];
        }

        // Batch norm weights are not needed anymore.
        bn_stddivs = null;
        bn_means = null;
        bn_betas = null;
        bn_gammas = null;
      }

    }

    public record SEunit
    {
      public SEunit(LC0LegacyWeights parent, Pblczero.Weights.SEunit se)
      {
        w1 = parent.WeightsDecoded(se.W1);
        b1 = parent.WeightsDecoded(se.B1);
        w2 = parent.WeightsDecoded(se.W2);
        b2 = parent.WeightsDecoded(se.B2);
      }

      public float[] w1;
      public float[] b1;
      public float[] w2;
      public float[] b2;
    }

    public struct MHA
    {
      public MHA(LC0LegacyWeights parent, Pblczero.Weights.Mha mha)
      {
        q_w = parent.WeightsDecoded(mha.QW);
        q_b = parent.WeightsDecoded(mha.QB);
        k_w = parent.WeightsDecoded(mha.KW);
        k_b = parent.WeightsDecoded(mha.KB);
        v_w = parent.WeightsDecoded(mha.VW);
        v_b = parent.WeightsDecoded(mha.VB);
        dense_w = parent.WeightsDecoded(mha.DenseW);
        dense_b = parent.WeightsDecoded(mha.DenseB);
      }

      public float[] q_w;
      public float[] q_b;
      public float[] k_w;
      public float[] k_b;
      public float[] v_w;
      public float[] v_b;
      public float[] dense_w;
      public float[] dense_b;
    }

    public struct FFN
    {
      public FFN(LC0LegacyWeights parent, Pblczero.Weights.Ffn ffn)
      {
        dense1_w = parent.WeightsDecoded(ffn.Dense1W);
        dense1_b = parent.WeightsDecoded(ffn.Dense1B);
        dense2_w = parent.WeightsDecoded(ffn.Dense2W);
        dense2_b = parent.WeightsDecoded(ffn.Dense2B);
      }

      public float[] dense1_w;
      public float[] dense1_b;
      public float[] dense2_w;
      public float[] dense2_b;
    }

    public struct EncoderLayer
    {
      public EncoderLayer(LC0LegacyWeights parent, Pblczero.Weights.EncoderLayer encoder)
      {
        mha = new MHA(parent, encoder.Mha);
        ln1_gammas = parent.WeightsDecoded(encoder.Ln1Gammas);
        ln1_betas = parent.WeightsDecoded(encoder.Ln1Betas);
        ffn = new FFN(parent, encoder.Ffn);
        ln2_gammas = parent.WeightsDecoded(encoder.Ln2Gammas);
        ln2_betas = parent.WeightsDecoded(encoder.Ln2Betas);
      }

      public MHA mha;
      public float[] ln1_gammas;
      public float[] ln1_betas;
      public FFN ffn;
      public float[] ln2_gammas;
      public float[] ln2_betas;
    }


    public struct Residual
    {
      public Residual(LC0LegacyWeights parent, Pblczero.Weights.Residual residual)
      {
        conv1 = new ConvBlock(parent, residual.Conv1);
        conv2 = new ConvBlock(parent, residual.Conv2);
        has_se = residual.Se != null;
        if (has_se)
        {
          se = new SEunit(parent, residual.Se);
        }
        else
        {
          se = default;
        }
      }

      public ConvBlock conv1;
      public ConvBlock conv2;
      public SEunit se;
      public bool has_se;
    }

    // Input convnet.
    public ConvBlock input;

    // Residual tower.
    public Residual[] residual;

    // Policy head
    // Extra convolution for AZ-style policy head
    public ConvBlock policy1;
    public ConvBlock policy;
    public float[] ip_pol_w;
    public float[] ip_pol_b;
    public float[] ip2_pol_w;
    public float[] ip2_pol_b;
    public float[] ip3_pol_w;
    public float[] ip3_pol_b;
    public float[] ip4_pol_w;

    public int numPolicyEncoderHeads; // pol_encoder_head_count;
    public EncoderLayer[] policyEncoders;

    // Value head
    public ConvBlock value;
    public float[] ip1_val_w;
    public float[] ip1_val_b;
    public float[] ip2_val_w;
    public float[] ip2_val_b;

#if ACTION_HEAD
    // Action head (experimental)
    public ConvBlock action;
    public float[] ip1_action_w;
    public float[] ip1_action_b;
#endif

    // Moves left head
    public ConvBlock moves_left;
    public float[] ip1_mov_w;
    public float[] ip1_mov_b;
    public float[] ip2_mov_w;
    public float[] ip2_mov_b;
    };
  }




