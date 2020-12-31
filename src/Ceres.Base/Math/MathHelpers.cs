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
  /// <summary>
  /// Miscellaneous math helper methods
  /// </summary>
  public static class MathHelpers
  {
    #region Min/max

    public static int Min(int f1, int f2) => System.Math.Min(f1, f2);
    public static int Max(int f1, int f2) => System.Math.Max(f1, f2);

    public static int Min(int f1, int f2, int f3) => System.Math.Min(f1, System.Math.Min(f2, f3));
    public static int Max(int f1, int f2, int f3) => System.Math.Max(f1, System.Math.Max(f2, f3));

    public static float Min(float f1, float f2) => System.Math.Min(f1, f2);
    public static float Max(float f1, float f2) => System.Math.Max(f1, f2);

    public static float Min(float f1, float f2, float f3) => System.Math.Min(f1, System.Math.Min(f2, f3));
    public static float Max(float f1, float f2, float f3) => System.Math.Max(f1, System.Math.Max(f2, f3));

    #endregion

    #region Bounds

    /// <summary>
    /// Returns the specified value, truncated to lie in the range [min, max].
    /// </summary>
    /// <param name="value"></param>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <returns></returns>
    public static float Bounded(float value, float min, float max)
    {
      if (value < min)
        return min;
      else if (value > max)
        return max;
      else
        return value;
    }
    
    #endregion
  }

}



