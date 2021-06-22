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
using System.Collections.Generic;
using System.Globalization;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.LC0.Positions;

#endregion

namespace Ceres.Chess.LC0VerboseMoves
{
  /// <summary>
  /// Captures the parsed output corresponding to one of the moves 
  /// from the LC0 verobse move statistics output.
  /// </summary>
  public class LC0VerboseMoveStat
  {
    /// <summary>
    /// parent
    /// </summary>
    public LC0VerboseMoveStats Parent;

    /// <summary>
    /// If statistics were successfully read
    /// </summary>
    public bool Valid;

    /// <summary>
    /// Move string for this move
    /// </summary>
    public string MoveString;

    /// <summary>
    /// Index of this move in the Leela encoding
    /// </summary>
    public int MoveCode;

    /// <summary>
    /// Number of visits during MCTS search
    /// </summary>
    public int VisitCount;

    /// <summary>
    /// Number of visits current in process
    /// </summary>
    public int VisitInFlightCount;

    /// <summary>
    /// Neural network policy forecast (% probability of playing this move),
    /// possibly with Dirichlet noise applied.
    /// </summary>
    public float P; 

    /// <summary>
    /// Average value of P in MCTS subtree
    /// </summary>
    public EncodedEvalLogistic Q;

    /// <summary>
    /// Win minus loss probability
    /// </summary>
    public float WL;

    /// <summary>
    /// Draw probability
    /// </summary>
    public float D;

    /// <summary>
    /// UCT exploration value
    /// </summary>
    public float U;

    /// <summary>
    /// Moves left
    /// </summary>
    public float M;

    /// <summary>
    /// Neural network output position evaluation for this position
    /// </summary>
    public EncodedEvalLogistic V;


    /// <summary>
    /// Returns string representation.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<LC0VerboseMoveStats  { MoveString } Visits={VisitCount} V = {V.LogisticValue} P={P} Q={Q.LogisticValue}, U={U} [{MoveCode}]";
    }


    /// <summary>
    /// Constructor (from the raw output line of LC0).
    /// </summary>
    /// <param name="parent"></param>
    /// <param name="line"></param>
    public LC0VerboseMoveStat(LC0VerboseMoveStats parent, string line)
    {
      Parent = parent;

      if (line != null)
      {
        Valid = ParseLine(line);
      }
      else
      {
        Valid = true;
      }
    }


    /// <summary>
    /// Returns the move described by this line.
    /// </summary>
    public Move Move => Move.FromUCI(MoveString);


    /// <summary>
    /// Parses the LC0 output line.
    /// </summary>
    /// <param name="line"></param>
    /// <returns></returns>
    bool ParseLine(string line)
    {
      // info string e2e4  (322 ) N:   15347 (+ 0) (V:   3.96%) (P: 10.09%) (Q:  0.03780) (U: 0.00707) (Q+U:  0.04487)
      // info string c2c4(264 ) N: 30259(+0)(V: 4.38 %)(P: 14.09 %)(Q: 0.03985)(U: 0.00501)(Q + U:  0.04486)

      string[] split = line.Split(new char[] { ' ', '(', ')', });
      if (split[0] != "info" || split[1] != "string")
      {
        return false;
      }

      MoveString = split[2];

      int nextPos = 3;
      double GetNextNum()
      {
        do
        {
          if (nextPos >= split.Length) return double.NaN;
          while (nextPos < split.Length && split[nextPos] == "") nextPos++;

          if (double.TryParse(split[nextPos++].Replace("%", ""), NumberStyles.Any, CultureInfo.InvariantCulture, out double val))
            return val;
        } while(true);
      }


      // info string e2e4(322 ) N: 15347(+0)(V: 3.96 %)(P: 10.09 %)(Q: 0.03780)(U: 0.00707)(Q + U:  0.04487)
      //new      info string d2d3(288 ) N: 11(+2)(P: 2.85 %)(WL: -0.00802)(D: 0.315)(M: 134.9)(Q: -0.00802)(U: 0.15138)(S: 0.14336)(V: 0.0092)

      MoveCode = (int)GetNextNum();
      nextPos++;

      VisitCount = (int)GetNextNum();
      VisitInFlightCount = (int)GetNextNum();

      Dictionary<string, float> stats = ExtractLC0VerboseMoveStats(line);

      stats.TryGetValue("P", out P);

      stats.TryGetValue("WL", out WL);

      stats.TryGetValue("D", out D);
      stats.TryGetValue("M", out M);

      float rawQ;
      stats.TryGetValue("Q", out rawQ);
      Q = EncodedEvalLogistic.FromLogistic(rawQ);

      stats.TryGetValue("D", out D);
      stats.TryGetValue("M", out M);

      stats.TryGetValue("U", out U);

      float S;
      stats.TryGetValue("S", out S);

      float rawV;
      stats.TryGetValue("V", out rawV);
      V = EncodedEvalLogistic.FromLogistic(rawV);

      return true;
    }


    /// <summary>
    // Parses out key/value pairs from Leela lines from --verbose-move-stats, such as:
    // "info string a1d4  (1607) N:  204283 (+71) (P: 14.76%) (WL:  0.29644) (D:  0.686) (Q:  0.29644) (U: 0.00419) (Q+U:  0.30064) (V:  0.1275)"
    /// </summary>
    /// <param name="statsLine"></param>
    /// <returns></returns>
    static Dictionary<string, float> ExtractLC0VerboseMoveStats(string statsLine)
    {
      Dictionary<string, float> ret = new Dictionary<string, float>();

      int curIndex = 0;
      while (true)
      {
        curIndex = statsLine.IndexOf(":", curIndex + 1);
        if (curIndex == -1)
        {
          break;
        }

        // backup to get label
        int backIndex = curIndex - 1;
        while (backIndex >= 0 && char.IsLetter(statsLine[backIndex]))
        {
          backIndex--;
        }

        string label = statsLine.Substring(backIndex + 1, curIndex - backIndex - 1);

        // advance past spaces
        curIndex++;
        while (curIndex < statsLine.Length && statsLine[curIndex] == ' ')
        {
          curIndex++;
        }

        // Extract numeric value
        string valueStr = "";
        while (curIndex < statsLine.Length && (char.IsDigit(statsLine[curIndex]) || statsLine[curIndex] == '.') || statsLine[curIndex] == '-')
        {
          valueStr += statsLine[curIndex++];
        }

        if (float.TryParse(valueStr, NumberStyles.Any, CultureInfo.InvariantCulture, out float value))
        {
          ret[label] = value;
        }
      }

      return ret;
    }


    /// <summary>
    /// Returns a string in same format as used by Leela Chess Zero.
    /// </summary>
    public string LC0String
    {
      get
      {
        // TODO: difference between WL and V (related to contempt?).
        return $"info string {MoveString}  ({MoveCode,3:F0} ) N:    {VisitCount,11:F0} (+{VisitInFlightCount,5:F0}) (P: {P,9:F2}%) "
             + $"(WL:  {V.LogisticValue,9:F4}) (D: {D,9:F4}) (M: {M,6:F1}) (Q:  {Q.LogisticValue,9:F4}) "
             + $"(U: {U,9:F4}) (V: {V.LogisticValue,9:F4})";
      }
    }

  }
}
