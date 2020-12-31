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


#endregion

namespace Ceres.Base.Math
{
  public static class ErrorFunction
  {
    /// <summary>
    /// Returns the value of the gaussian error function at <paramref name="x"/>.
    /// </summary>
    public static double Erf(double x)
    {
      /*
      Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
      *
      * Developed at SunPro, a Sun Microsystems, Inc. business.
      * Permission to use, copy, modify, and distribute this
      * software is freely granted, provided that this notice
      * is preserved.
      */

      #region Constants

      const double tiny = 1e-300;
      const double erx = 8.45062911510467529297e-01;

      // Coefficients for approximation to erf on [0, 0.84375]
      const double efx = 1.28379167095512586316e-01; /* 0x3FC06EBA; 0x8214DB69 */
      const double efx8 = 1.02703333676410069053e+00; /* 0x3FF06EBA; 0x8214DB69 */
      const double pp0 = 1.28379167095512558561e-01; /* 0x3FC06EBA; 0x8214DB68 */
      const double pp1 = -3.25042107247001499370e-01; /* 0xBFD4CD7D; 0x691CB913 */
      const double pp2 = -2.84817495755985104766e-02; /* 0xBF9D2A51; 0xDBD7194F */
      const double pp3 = -5.77027029648944159157e-03; /* 0xBF77A291; 0x236668E4 */
      const double pp4 = -2.37630166566501626084e-05; /* 0xBEF8EAD6; 0x120016AC */
      const double qq1 = 3.97917223959155352819e-01; /* 0x3FD97779; 0xCDDADC09 */
      const double qq2 = 6.50222499887672944485e-02; /* 0x3FB0A54C; 0x5536CEBA */
      const double qq3 = 5.08130628187576562776e-03; /* 0x3F74D022; 0xC4D36B0F */
      const double qq4 = 1.32494738004321644526e-04; /* 0x3F215DC9; 0x221C1A10 */
      const double qq5 = -3.96022827877536812320e-06; /* 0xBED09C43; 0x42A26120 */

      // Coefficients for approximation to erf in [0.84375, 1.25]
      const double pa0 = -2.36211856075265944077e-03; /* 0xBF6359B8; 0xBEF77538 */
      const double pa1 = 4.14856118683748331666e-01; /* 0x3FDA8D00; 0xAD92B34D */
      const double pa2 = -3.72207876035701323847e-01; /* 0xBFD7D240; 0xFBB8C3F1 */
      const double pa3 = 3.18346619901161753674e-01; /* 0x3FD45FCA; 0x805120E4 */
      const double pa4 = -1.10894694282396677476e-01; /* 0xBFBC6398; 0x3D3E28EC */
      const double pa5 = 3.54783043256182359371e-02; /* 0x3FA22A36; 0x599795EB */
      const double pa6 = -2.16637559486879084300e-03; /* 0xBF61BF38; 0x0A96073F */
      const double qa1 = 1.06420880400844228286e-01; /* 0x3FBB3E66; 0x18EEE323 */
      const double qa2 = 5.40397917702171048937e-01; /* 0x3FE14AF0; 0x92EB6F33 */
      const double qa3 = 7.18286544141962662868e-02; /* 0x3FB2635C; 0xD99FE9A7 */
      const double qa4 = 1.26171219808761642112e-01; /* 0x3FC02660; 0xE763351F */
      const double qa5 = 1.36370839120290507362e-02; /* 0x3F8BEDC2; 0x6B51DD1C */
      const double qa6 = 1.19844998467991074170e-02; /* 0x3F888B54; 0x5735151D */

      // Coefficients for approximation to erfc in [1.25, 1/0.35]
      const double ra0 = -9.86494403484714822705e-03; /* 0xBF843412; 0x600D6435 */
      const double ra1 = -6.93858572707181764372e-01; /* 0xBFE63416; 0xE4BA7360 */
      const double ra2 = -1.05586262253232909814e+01; /* 0xC0251E04; 0x41B0E726 */
      const double ra3 = -6.23753324503260060396e+01; /* 0xC04F300A; 0xE4CBA38D */
      const double ra4 = -1.62396669462573470355e+02; /* 0xC0644CB1; 0x84282266 */
      const double ra5 = -1.84605092906711035994e+02; /* 0xC067135C; 0xEBCCABB2 */
      const double ra6 = -8.12874355063065934246e+01; /* 0xC0545265; 0x57E4D2F2 */
      const double ra7 = -9.81432934416914548592e+00; /* 0xC023A0EF; 0xC69AC25C */
      const double sa1 = 1.96512716674392571292e+01; /* 0x4033A6B9; 0xBD707687 */
      const double sa2 = 1.37657754143519042600e+02; /* 0x4061350C; 0x526AE721 */
      const double sa3 = 4.34565877475229228821e+02; /* 0x407B290D; 0xD58A1A71 */
      const double sa4 = 6.45387271733267880336e+02; /* 0x40842B19; 0x21EC2868 */
      const double sa5 = 4.29008140027567833386e+02; /* 0x407AD021; 0x57700314 */
      const double sa6 = 1.08635005541779435134e+02; /* 0x405B28A3; 0xEE48AE2C */
      const double sa7 = 6.57024977031928170135e+00; /* 0x401A47EF; 0x8E484A93 */
      const double sa8 = -6.04244152148580987438e-02; /* 0xBFAEEFF2; 0xEE749A62 */

      // Coefficients for approximation to erfc in [1/0.35, 28]
      const double rb0 = -9.86494292470009928597e-03; /* 0xBF843412; 0x39E86F4A */
      const double rb1 = -7.99283237680523006574e-01; /* 0xBFE993BA; 0x70C285DE */
      const double rb2 = -1.77579549177547519889e+01; /* 0xC031C209; 0x555F995A */
      const double rb3 = -1.60636384855821916062e+02; /* 0xC064145D; 0x43C5ED98 */
      const double rb4 = -6.37566443368389627722e+02; /* 0xC083EC88; 0x1375F228 */
      const double rb5 = -1.02509513161107724954e+03; /* 0xC0900461; 0x6A2E5992 */
      const double rb6 = -4.83519191608651397019e+02; /* 0xC07E384E; 0x9BDC383F */
      const double sb1 = 3.03380607434824582924e+01; /* 0x403E568B; 0x261D5190 */
      const double sb2 = 3.25792512996573918826e+02; /* 0x40745CAE; 0x221B9F0A */
      const double sb3 = 1.53672958608443695994e+03; /* 0x409802EB; 0x189D5118 */
      const double sb4 = 3.19985821950859553908e+03; /* 0x40A8FFB7; 0x688C246A */
      const double sb5 = 2.55305040643316442583e+03; /* 0x40A3F219; 0xCEDF3BE6 */
      const double sb6 = 4.74528541206955367215e+02; /* 0x407DA874; 0xE79FE763 */
      const double sb7 = -2.24409524465858183362e+01; /* 0xC03670E2; 0x42712D62 */

      #endregion

      if (double.IsNaN(x))
        return double.NaN;

      if (double.IsNegativeInfinity(x))
        return -1.0;

      if (double.IsPositiveInfinity(x))
        return 1.0;

      int n0, hx, ix;
      double R, S, P, Q, s, y, z, r;
      unsafe
      {
        double one = 1.0;
        n0 = ((*(int*)&one) >> 29) ^ 1;
        hx = *(n0 + (int*)&x);
      }
      ix = hx & 0x7FFFFFFF;

      if (ix < 0x3FEB0000) // |x| < 0.84375
      {
        if (ix < 0x3E300000) // |x| < 2**-28
        {
          if (ix < 0x00800000)
            return 0.125 * (8.0 * x + efx8 * x); // avoid underflow
          return x + efx * x;
        }
        z = x * x;
        r = pp0 + z * (pp1 + z * (pp2 + z * (pp3 + z * pp4)));
        s = 1.0 + z * (qq1 + z * (qq2 + z * (qq3 + z * (qq4 + z * qq5))));
        y = r / s;
        return x + x * y;
      }
      if (ix < 0x3FF40000) // 0.84375 <= |x| < 1.25
      {
        s = System.Math.Abs(x) - 1.0;
        P = pa0 + s * (pa1 + s * (pa2 + s * (pa3 + s * (pa4 + s * (pa5 + s * pa6)))));
        Q = 1.0 + s * (qa1 + s * (qa2 + s * (qa3 + s * (qa4 + s * (qa5 + s * qa6)))));
        if (hx >= 0)
          return erx + P / Q;
        else
          return -erx - P / Q;
      }
      if (ix >= 0x40180000) // inf > |x| >= 6
      {
        if (hx >= 0)
          return 1.0 - tiny;
        else
          return tiny - 1.0;
      }
      x = System.Math.Abs(x);
      s = 1.0 / (x * x);
      if (ix < 0x4006DB6E) // |x| < 1/0.35
      {
        R = ra0 + s * (ra1 + s * (ra2 + s * (ra3 + s * (ra4 + s * (ra5 + s * (ra6 + s * ra7))))));
        S = 1.0 + s * (sa1 + s * (sa2 + s * (sa3 + s * (sa4 + s * (sa5 + s * (sa6 + s * (sa7 + s * sa8)))))));
      }
      else // |x| >= 1/0.35
      {
        R = rb0 + s * (rb1 + s * (rb2 + s * (rb3 + s * (rb4 + s * (rb5 + s * rb6)))));
        S = 1.0 + s * (sb1 + s * (sb2 + s * (sb3 + s * (sb4 + s * (sb5 + s * (sb6 + s * sb7))))));
      }
      z = x;
      unsafe { *(1 - n0 + (int*)&z) = 0; }
      r = System.Math.Exp(-z * z - 0.5625) * System.Math.Exp((z - x) * (z + x) + R / S);
      if (hx >= 0)
        return 1.0 - r / x;
      else
        return r / x - 1.0;
    }


    /// <summary>
    /// Returns the value of the complementary error function at <paramref name="x"/>.
    /// </summary>
    public static double Erfc(double x)
    {
      /*
      Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
      *
      * Developed at SunPro, a Sun Microsystems, Inc. business.
      * Permission to use, copy, modify, and distribute this
      * software is freely granted, provided that this notice
      * is preserved.
      */

      #region Constants

      const double tiny = 1e-300;
      const double erx = 8.45062911510467529297e-01;

      // Coefficients for approximation to erf on [0, 0.84375]
      const double pp0 = 1.28379167095512558561e-01; /* 0x3FC06EBA; 0x8214DB68 */
      const double pp1 = -3.25042107247001499370e-01; /* 0xBFD4CD7D; 0x691CB913 */
      const double pp2 = -2.84817495755985104766e-02; /* 0xBF9D2A51; 0xDBD7194F */
      const double pp3 = -5.77027029648944159157e-03; /* 0xBF77A291; 0x236668E4 */
      const double pp4 = -2.37630166566501626084e-05; /* 0xBEF8EAD6; 0x120016AC */
      const double qq1 = 3.97917223959155352819e-01; /* 0x3FD97779; 0xCDDADC09 */
      const double qq2 = 6.50222499887672944485e-02; /* 0x3FB0A54C; 0x5536CEBA */
      const double qq3 = 5.08130628187576562776e-03; /* 0x3F74D022; 0xC4D36B0F */
      const double qq4 = 1.32494738004321644526e-04; /* 0x3F215DC9; 0x221C1A10 */
      const double qq5 = -3.96022827877536812320e-06; /* 0xBED09C43; 0x42A26120 */

      // Coefficients for approximation to erf in [0.84375, 1.25]
      const double pa0 = -2.36211856075265944077e-03; /* 0xBF6359B8; 0xBEF77538 */
      const double pa1 = 4.14856118683748331666e-01; /* 0x3FDA8D00; 0xAD92B34D */
      const double pa2 = -3.72207876035701323847e-01; /* 0xBFD7D240; 0xFBB8C3F1 */
      const double pa3 = 3.18346619901161753674e-01; /* 0x3FD45FCA; 0x805120E4 */
      const double pa4 = -1.10894694282396677476e-01; /* 0xBFBC6398; 0x3D3E28EC */
      const double pa5 = 3.54783043256182359371e-02; /* 0x3FA22A36; 0x599795EB */
      const double pa6 = -2.16637559486879084300e-03; /* 0xBF61BF38; 0x0A96073F */
      const double qa1 = 1.06420880400844228286e-01; /* 0x3FBB3E66; 0x18EEE323 */
      const double qa2 = 5.40397917702171048937e-01; /* 0x3FE14AF0; 0x92EB6F33 */
      const double qa3 = 7.18286544141962662868e-02; /* 0x3FB2635C; 0xD99FE9A7 */
      const double qa4 = 1.26171219808761642112e-01; /* 0x3FC02660; 0xE763351F */
      const double qa5 = 1.36370839120290507362e-02; /* 0x3F8BEDC2; 0x6B51DD1C */
      const double qa6 = 1.19844998467991074170e-02; /* 0x3F888B54; 0x5735151D */

      // Coefficients for approximation to erfc in [1.25, 1/0.35]
      const double ra0 = -9.86494403484714822705e-03; /* 0xBF843412; 0x600D6435 */
      const double ra1 = -6.93858572707181764372e-01; /* 0xBFE63416; 0xE4BA7360 */
      const double ra2 = -1.05586262253232909814e+01; /* 0xC0251E04; 0x41B0E726 */
      const double ra3 = -6.23753324503260060396e+01; /* 0xC04F300A; 0xE4CBA38D */
      const double ra4 = -1.62396669462573470355e+02; /* 0xC0644CB1; 0x84282266 */
      const double ra5 = -1.84605092906711035994e+02; /* 0xC067135C; 0xEBCCABB2 */
      const double ra6 = -8.12874355063065934246e+01; /* 0xC0545265; 0x57E4D2F2 */
      const double ra7 = -9.81432934416914548592e+00; /* 0xC023A0EF; 0xC69AC25C */
      const double sa1 = 1.96512716674392571292e+01; /* 0x4033A6B9; 0xBD707687 */
      const double sa2 = 1.37657754143519042600e+02; /* 0x4061350C; 0x526AE721 */
      const double sa3 = 4.34565877475229228821e+02; /* 0x407B290D; 0xD58A1A71 */
      const double sa4 = 6.45387271733267880336e+02; /* 0x40842B19; 0x21EC2868 */
      const double sa5 = 4.29008140027567833386e+02; /* 0x407AD021; 0x57700314 */
      const double sa6 = 1.08635005541779435134e+02; /* 0x405B28A3; 0xEE48AE2C */
      const double sa7 = 6.57024977031928170135e+00; /* 0x401A47EF; 0x8E484A93 */
      const double sa8 = -6.04244152148580987438e-02; /* 0xBFAEEFF2; 0xEE749A62 */

      // Coefficients for approximation to erfc in [1/0.35, 28]
      const double rb0 = -9.86494292470009928597e-03; /* 0xBF843412; 0x39E86F4A */
      const double rb1 = -7.99283237680523006574e-01; /* 0xBFE993BA; 0x70C285DE */
      const double rb2 = -1.77579549177547519889e+01; /* 0xC031C209; 0x555F995A */
      const double rb3 = -1.60636384855821916062e+02; /* 0xC064145D; 0x43C5ED98 */
      const double rb4 = -6.37566443368389627722e+02; /* 0xC083EC88; 0x1375F228 */
      const double rb5 = -1.02509513161107724954e+03; /* 0xC0900461; 0x6A2E5992 */
      const double rb6 = -4.83519191608651397019e+02; /* 0xC07E384E; 0x9BDC383F */
      const double sb1 = 3.03380607434824582924e+01; /* 0x403E568B; 0x261D5190 */
      const double sb2 = 3.25792512996573918826e+02; /* 0x40745CAE; 0x221B9F0A */
      const double sb3 = 1.53672958608443695994e+03; /* 0x409802EB; 0x189D5118 */
      const double sb4 = 3.19985821950859553908e+03; /* 0x40A8FFB7; 0x688C246A */
      const double sb5 = 2.55305040643316442583e+03; /* 0x40A3F219; 0xCEDF3BE6 */
      const double sb6 = 4.74528541206955367215e+02; /* 0x407DA874; 0xE79FE763 */
      const double sb7 = -2.24409524465858183362e+01; /* 0xC03670E2; 0x42712D62 */

      #endregion

      if (double.IsNaN(x))
        return double.NaN;

      if (double.IsNegativeInfinity(x))
        return 2.0;

      if (double.IsPositiveInfinity(x))
        return 0.0;

      int n0, hx, ix;
      double R, S, P, Q, s, y, z, r;
      unsafe
      {
        double one = 1.0;
        n0 = ((*(int*)&one) >> 29) ^ 1;
        hx = *(n0 + (int*)&x);
      }
      ix = hx & 0x7FFFFFFF;

      if (ix < 0x3FEB0000) // |x| < 0.84375
      {
        if (ix < 0x3C700000) // |x| < 2**-56
          return 1.0 - x;
        z = x * x;
        r = pp0 + z * (pp1 + z * (pp2 + z * (pp3 + z * pp4)));
        s = 1.0 + z * (qq1 + z * (qq2 + z * (qq3 + z * (qq4 + z * qq5))));
        y = r / s;
        if (hx < 0x3FD00000) // x < 1/4
          return 1.0 - (x + x * y);
        else
        {
          r = x * y;
          r += (x - 0.5);
          return 0.5 - r;
        }
      }
      if (ix < 0x3FF40000) // 0.84375 <= |x| < 1.25
      {
        s = System.Math.Abs(x) - 1.0;
        P = pa0 + s * (pa1 + s * (pa2 + s * (pa3 + s * (pa4 + s * (pa5 + s * pa6)))));
        Q = 1.0 + s * (qa1 + s * (qa2 + s * (qa3 + s * (qa4 + s * (qa5 + s * qa6)))));
        if (hx >= 0)
        {
          z = 1.0 - erx;
          return z - P / Q;
        }
        else
        {
          z = erx + P / Q;
          return 1.0 + z;
        }
      }
      if (ix < 0x403C0000) // |x| < 28
      {
        x = System.Math.Abs(x);
        s = 1.0 / (x * x);
        if (ix < 0x4006DB6D) // |x| < 1/.35 ~ 2.857143
        {
          R = ra0 + s * (ra1 + s * (ra2 + s * (ra3 + s * (ra4 + s * (ra5 + s * (ra6 + s * ra7))))));
          S = 1.0 + s * (sa1 + s * (sa2 + s * (sa3 + s * (sa4 + s * (sa5 + s * (sa6 + s * (sa7 + s * sa8)))))));
        }
        else // |x| >= 1/.35 ~ 2.857143
        {
          if (hx < 0 && ix >= 0x40180000)
            return 2.0 - tiny; // x < -6
          R = rb0 + s * (rb1 + s * (rb2 + s * (rb3 + s * (rb4 + s * (rb5 + s * rb6)))));
          S = 1.0 + s * (sb1 + s * (sb2 + s * (sb3 + s * (sb4 + s * (sb5 + s * (sb6 + s * sb7))))));
        }
        z = x;
        unsafe { *(1 - n0 + (int*)&z) = 0; }
        r = System.Math.Exp(-z * z - 0.5625) *
        System.Math.Exp((z - x) * (z + x) + R / S);
        if (hx > 0)
          return r / x;
        else
          return 2.0 - r / x;
      }
      else
      {
        if (hx > 0)
          return tiny * tiny;
        else
          return 2.0 - tiny;
      }
    }

  }
}
