#region License notice

// NOTE: This file is substantially a transliteration from C to C# 
//       of code from the Fathom project.
//       Both Fathom and Ceres copyrights are included below.

/*
Copyright (c) 2015 basil00
Modifications Copyright (c) 2016-2020 by Jon Dart
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

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
using System.Diagnostics;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Text;
using Ceres.Base.OperatingSystem;

using static Ceres.Chess.TBBackends.Fathom.FathomMoveGen;

#endregion

namespace Ceres.Chess.TBBackends.Fathom
{

  internal class FathomProbe
  {
    const bool VERBOSE = false;

    public string paths;

    const int TB_HASHBITS = (TB_PIECES < 7 ? 11 : 12);
    const int TB_MAX_PIECE = (TB_PIECES < 7 ? 254 : 650);
    const int TB_MAX_PAWN = (TB_PIECES < 7 ? 256 : 861);
    const int TB_MAX_SYMS = 4096;

    /***************************************************************************/
    /* SCORING CONSTANTS                                                       */
    /***************************************************************************/
    /*
     * Fathom can produce scores for tablebase moves. These depend on the
     * value of a pawn, and the magnitude of mate scores. The following
     * constants are representative values but will likely need
     * modification to adapt to an engine's own internal score values.
     */
    const int TB_VALUE_PAWN = 100; /* value of pawn in endgame */
    const int TB_VALUE_MATE = 32000;
    const int TB_VALUE_INFINITE = 32767; /* value above all normal score values */
    const int TB_VALUE_DRAW = 0;
    const int TB_MAX_MATE_PLY = 255;



    int TB_MaxCardinality = 0;
    int TB_MaxCardinalityDTM = 0;
    internal int TB_LARGEST = 0;
    //extern int TB_CardinalityDTM;

    static readonly string[] tbSuffix = { ".rtbw", ".rtbm", ".rtbz" };
    static uint[] tbMagic = { 0x5d23e871, 0x88ac504b, 0xa50c66d7 };


    enum PieceFileRank { PIECE_ENC, FILE_ENC, RANK_ENC };

    static string SEP_CHAR => SoftwareManager.IsLinux ? ":" : ";"; // TODO: possibly use native .NET here ?

    static T[] CreateInitializedArray<T>(int count, Action<T> initializer = null) where T : class, new()
    {

      T[] ret = new T[count];
      for (int i = 0; i < ret.Length; i++)
      {
        T newT = new T();
        ret[i] = newT;
        initializer?.Invoke(newT);
      }

      return ret;
    }

    internal void tb_free()
    {
      foreach (PairsDataPtr v in allocatedPairsData)
      {
        v.Free();
      }
      allocatedPairsData.Clear();
    }

    internal unsafe bool tb_init(string paths)
    {
      if (paths == null)
      {
        throw new Exception("Null tablebase path is not supported.");
      }

      if (!staticsInitialized)
      {
        init_indices();
        FathomMoveGen.Init();
        staticsInitialized = true;
      }

      this.paths = paths;

#if REINITIALIZATION_NOT_SUPPORTED
    // if pathString is set, we need to clean up first.
    if (pathString)
    {
      free(pathString);
      free(paths);

      for (int i = 0; i < tbNumPiece; i++)
        free_tb_entry((struct BaseEntry *)&pieceEntry[i]);
    for (int i = 0; i<tbNumPawn; i++)
      free_tb_entry((struct BaseEntry *)&pawnEntry[i]);

    LOCK_DESTROY(tbMutex);

    pathString = NULL;
    numWdl = numDtm = numDtz = 0;
  }
#endif

      TB_LARGEST = 0;

      // if path is an empty string or equals "<empty>", we are done.
      if (paths == null || paths == "<empty>")
      {
        return true;
      }


      //pathsArray = path.Split(SEP_CHAR);
#if NOT
    string pathString;
    pathString = (char*)malloc(strlen(p) + 1);
    strcpy(pathString, p);
    int numPaths = 0;
    for (int i = 0; ; i++)
    {
      if (pathString[i] != SEP_CHAR)
        numPaths++;
      while (pathString[i] && pathString[i] != SEP_CHAR)
        i++;
      if (!pathString[i]) break;
      pathString[i] = 0;
    }
    paths = (char**)malloc(numPaths * sizeof(*paths));
    for (int i = 0, j = 0; i < numPaths; i++)
    {
      while (!pathString[j]) j++;
      paths[i] = &pathString[j];
      while (pathString[j]) j++;
    }

    LOCK_INIT(tbMutex);
#endif

      int tbNumPiece;
      int tbNumPawn = 0;
      TB_MaxCardinality = TB_MaxCardinalityDTM = 0;


      if (pieceEntry == null)
      {
        pieceEntry = CreateInitializedArray<PieceEntry>(TB_MAX_PIECE);
        pawnEntry = CreateInitializedArray<PawnEntry>(TB_MAX_PAWN);
      }


      for (int i = 0; i < (1 << TB_HASHBITS); i++)
      {
        tbHash[i].key = 0;
        tbHash[i].ptr = null;
      }

      for (int i = 0; i < 5; i++)
      {
        init_tb($"K{pchr(i)}vK");
      }

      for (int i = 0; i < 5; i++)
      {
        for (int j = i; j < 5; j++)
        {
          init_tb($"K{pchr(i)}vK{pchr(j)}");
        }
      }

      for (int i = 0; i < 5; i++)
      {
        for (int j = i; j < 5; j++)
        {
          init_tb($"K{pchr(i)}{pchr(j)}vK");
        }
      }

      for (int i = 0; i < 5; i++)
      {
        for (int j = i; j < 5; j++)
        {
          for (int k = 0; k < 5; k++)
          {
            init_tb($"K{pchr(i)}{pchr(j)}vK{pchr(k)}");
          }
        }
      }

      for (int i = 0; i < 5; i++)
      {
        for (int j = i; j < 5; j++)
        {
          for (int k = j; k < 5; k++)
          {
            init_tb($"K{pchr(i)}{pchr(j)}{pchr(k)}vK");
          }
        }
      }

      for (int i = 0; i < 5; i++)
      {
        for (int j = i; j < 5; j++)
        {
          for (int k = i; k < 5; k++)
          {
            for (int l = (i == k) ? j : k; l < 5; l++)
            {
              init_tb($"K{pchr(i)}{pchr(j)}vK{pchr(k)}{pchr(l)}");
            }
          }
        }
      }

      for (int i = 0; i < 5; i++)
      {
        for (int j = i; j < 5; j++)
        {
          for (int k = j; k < 5; k++)
          {
            for (int l = 0; l < 5; l++)
            {
              init_tb($"K{pchr(i)}{pchr(j)}{pchr(k)}vK{pchr(l)}");
            }
          }
        }
      }

      for (int i = 0; i < 5; i++)
      {
        for (int j = i; j < 5; j++)
        {
          for (int k = j; k < 5; k++)
          {
            for (int l = k; l < 5; l++)
            {
              init_tb($"K{pchr(i)}{pchr(j)}{pchr(k)}{pchr(l)}vK");
            }
          }
        }
      }

      if (TB_PIECES >= 7)
      {

        for (int i = 0; i < 5; i++)
        {
          for (int j = i; j < 5; j++)
          {
            for (int k = j; k < 5; k++)
            {
              for (int l = k; l < 5; l++)
              {
                for (int m = l; m < 5; m++)
                {
                  init_tb($"K{pchr(i)}{pchr(j)}{pchr(k)}{pchr(l)}{pchr(m)}vK");
                }
              }
            }
          }
        }

        for (int i = 0; i < 5; i++)
        {
          for (int j = i; j < 5; j++)
          {
            for (int k = j; k < 5; k++)
            {
              for (int l = k; l < 5; l++)
              {
                for (int m = 0; m < 5; m++)
                {
                  init_tb($"K{pchr(i)}{pchr(j)}{pchr(k)}{pchr(l)}vK{pchr(m)}");
                }
              }
            }
          }
        }

        for (int i = 0; i < 5; i++)
        {
          for (int j = i; j < 5; j++)
          {
            for (int k = j; k < 5; k++)
            {
              for (int l = 0; l < 5; l++)
              {
                for (int m = l; m < 5; m++)
                {
                  init_tb($"K{pchr(i)}{pchr(j)}{pchr(k)}vK{pchr(l)}{pchr(m)}");
                }
              }
            }
          }
        }
      }


      /* TBD - assumes UCI
      printf("info string Found %d WDL, %d DTM and %d DTZ tablebase files.\n",  numWdl, numDtm, numDtz);
      fflush(stdout);
      */

      // Set TB_LARGEST, for backward compatibility with pre-7-man Fathom
      TB_LARGEST = (int)TB_MaxCardinality;
      if ((int)TB_MaxCardinalityDTM > TB_LARGEST)
      {
        TB_LARGEST = TB_MaxCardinalityDTM;
      }
      return true;
    }


    static readonly sbyte[] OffDiag = new sbyte[]
    {
  0,-1,-1,-1,-1,-1,-1,-1,
  1, 0,-1,-1,-1,-1,-1,-1,
  1, 1, 0,-1,-1,-1,-1,-1,
  1, 1, 1, 0,-1,-1,-1,-1,
  1, 1, 1, 1, 0,-1,-1,-1,
  1, 1, 1, 1, 1, 0,-1,-1,
  1, 1, 1, 1, 1, 1, 0,-1,
  1, 1, 1, 1, 1, 1, 1, 0
  };

    static readonly byte[] Triangle = new byte[]
    {
  6, 0, 1, 2, 2, 1, 0, 6,
  0, 7, 3, 4, 4, 3, 7, 0,
  1, 3, 8, 5, 5, 8, 3, 1,
  2, 4, 5, 9, 9, 5, 4, 2,
  2, 4, 5, 9, 9, 5, 4, 2,
  1, 3, 8, 5, 5, 8, 3, 1,
  0, 7, 3, 4, 4, 3, 7, 0,
  6, 0, 1, 2, 2, 1, 0, 6
  };

    static readonly byte[] FlipDiag = new byte[]
    {
   0,  8, 16, 24, 32, 40, 48, 56,
   1,  9, 17, 25, 33, 41, 49, 57,
   2, 10, 18, 26, 34, 42, 50, 58,
   3, 11, 19, 27, 35, 43, 51, 59,
   4, 12, 20, 28, 36, 44, 52, 60,
   5, 13, 21, 29, 37, 45, 53, 61,
   6, 14, 22, 30, 38, 46, 54, 62,
   7, 15, 23, 31, 39, 47, 55, 63
    };

    static readonly byte[] Lower = new byte[]
    {
  28,  0,  1,  2,  3,  4,  5,  6,
   0, 29,  7,  8,  9, 10, 11, 12,
   1,  7, 30, 13, 14, 15, 16, 17,
   2,  8, 13, 31, 18, 19, 20, 21,
   3,  9, 14, 18, 32, 22, 23, 24,
   4, 10, 15, 19, 22, 33, 25, 26,
   5, 11, 16, 20, 23, 25, 34, 27,
   6, 12, 17, 21, 24, 26, 27, 35
    };

    static readonly byte[] Diag = new byte[]
    {
   0,  0,  0,  0,  0,  0,  0,  8,
   0,  1,  0,  0,  0,  0,  9,  0,
   0,  0,  2,  0,  0, 10,  0,  0,
   0,  0,  0,  3, 11,  0,  0,  0,
   0,  0,  0, 12,  4,  0,  0,  0,
   0,  0, 13,  0,  0,  5,  0,  0,
   0, 14,  0,  0,  0,  0,  6,  0,
  15,  0,  0,  0,  0,  0,  0,  7
  };

    static readonly byte[,] Flap = new byte[,]
    {
  {  0,  0,  0,  0,  0,  0,  0,  0,
     0,  6, 12, 18, 18, 12,  6,  0,
     1,  7, 13, 19, 19, 13,  7,  1,
     2,  8, 14, 20, 20, 14,  8,  2,
     3,  9, 15, 21, 21, 15,  9,  3,
     4, 10, 16, 22, 22, 16, 10,  4,
     5, 11, 17, 23, 23, 17, 11,  5,
     0,  0,  0,  0,  0,  0,  0,  0  },
  {  0,  0,  0,  0,  0,  0,  0,  0,
     0,  1,  2,  3,  3,  2,  1,  0,
     4,  5,  6,  7,  7,  6,  5,  4,
     8,  9, 10, 11, 11, 10,  9,  8,
    12, 13, 14, 15, 15, 14, 13, 12,
    16, 17, 18, 19, 19, 18, 17, 16,
    20, 21, 22, 23, 23, 22, 21, 20,
     0,  0,  0,  0,  0,  0,  0,  0  }
  };

    static readonly byte[,] PawnTwist = new byte[,] { // 2,64
  {  0,  0,  0,  0,  0,  0,  0,  0,
    47, 35, 23, 11, 10, 22, 34, 46,
    45, 33, 21,  9,  8, 20, 32, 44,
    43, 31, 19,  7,  6, 18, 30, 42,
    41, 29, 17,  5,  4, 16, 28, 40,
    39, 27, 15,  3,  2, 14, 26, 38,
    37, 25, 13,  1,  0, 12, 24, 36,
     0,  0,  0,  0,  0,  0,  0,  0 },
  {  0,  0,  0,  0,  0,  0,  0,  0,
    47, 45, 43, 41, 40, 42, 44, 46,
    39, 37, 35, 33, 32, 34, 36, 38,
    31, 29, 27, 25, 24, 26, 28, 30,
    23, 21, 19, 17, 16, 18, 20, 22,
    15, 13, 11,  9,  8, 10, 12, 14,
     7,  5,  3,  1,  0,  2,  4,  6,
     0,  0,  0,  0,  0,  0,  0,  0 }
};

    static readonly short[,] KKIdx = new short[,]{ // 10,64
  { -1, -1, -1,  0,  1,  2,  3,  4,
    -1, -1, -1,  5,  6,  7,  8,  9,
    10, 11, 12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33,
    34, 35, 36, 37, 38, 39, 40, 41,
    42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57 },
  { 58, -1, -1, -1, 59, 60, 61, 62,
    63, -1, -1, -1, 64, 65, 66, 67,
    68, 69, 70, 71, 72, 73, 74, 75,
    76, 77, 78, 79, 80, 81, 82, 83,
    84, 85, 86, 87, 88, 89, 90, 91,
    92, 93, 94, 95, 96, 97, 98, 99,
   100,101,102,103,104,105,106,107,
   108,109,110,111,112,113,114,115},
  {116,117, -1, -1, -1,118,119,120,
   121,122, -1, -1, -1,123,124,125,
   126,127,128,129,130,131,132,133,
   134,135,136,137,138,139,140,141,
   142,143,144,145,146,147,148,149,
   150,151,152,153,154,155,156,157,
   158,159,160,161,162,163,164,165,
   166,167,168,169,170,171,172,173 },
  {174, -1, -1, -1,175,176,177,178,
   179, -1, -1, -1,180,181,182,183,
   184, -1, -1, -1,185,186,187,188,
   189,190,191,192,193,194,195,196,
   197,198,199,200,201,202,203,204,
   205,206,207,208,209,210,211,212,
   213,214,215,216,217,218,219,220,
   221,222,223,224,225,226,227,228 },
  {229,230, -1, -1, -1,231,232,233,
   234,235, -1, -1, -1,236,237,238,
   239,240, -1, -1, -1,241,242,243,
   244,245,246,247,248,249,250,251,
   252,253,254,255,256,257,258,259,
   260,261,262,263,264,265,266,267,
   268,269,270,271,272,273,274,275,
   276,277,278,279,280,281,282,283 },
  {284,285,286,287,288,289,290,291,
   292,293, -1, -1, -1,294,295,296,
   297,298, -1, -1, -1,299,300,301,
   302,303, -1, -1, -1,304,305,306,
   307,308,309,310,311,312,313,314,
   315,316,317,318,319,320,321,322,
   323,324,325,326,327,328,329,330,
   331,332,333,334,335,336,337,338 },
  { -1, -1,339,340,341,342,343,344,
    -1, -1,345,346,347,348,349,350,
    -1, -1,441,351,352,353,354,355,
    -1, -1, -1,442,356,357,358,359,
    -1, -1, -1, -1,443,360,361,362,
    -1, -1, -1, -1, -1,444,363,364,
    -1, -1, -1, -1, -1, -1,445,365,
    -1, -1, -1, -1, -1, -1, -1,446 },
  { -1, -1, -1,366,367,368,369,370,
    -1, -1, -1,371,372,373,374,375,
    -1, -1, -1,376,377,378,379,380,
    -1, -1, -1,447,381,382,383,384,
    -1, -1, -1, -1,448,385,386,387,
    -1, -1, -1, -1, -1,449,388,389,
    -1, -1, -1, -1, -1, -1,450,390,
    -1, -1, -1, -1, -1, -1, -1,451 },
  {452,391,392,393,394,395,396,397,
    -1, -1, -1, -1,398,399,400,401,
    -1, -1, -1, -1,402,403,404,405,
    -1, -1, -1, -1,406,407,408,409,
    -1, -1, -1, -1,453,410,411,412,
    -1, -1, -1, -1, -1,454,413,414,
    -1, -1, -1, -1, -1, -1,455,415,
    -1, -1, -1, -1, -1, -1, -1,456 },
  {457,416,417,418,419,420,421,422,
    -1,458,423,424,425,426,427,428,
    -1, -1, -1, -1, -1,429,430,431,
    -1, -1, -1, -1, -1,432,433,434,
    -1, -1, -1, -1, -1,435,436,437,
    -1, -1, -1, -1, -1,459,438,439,
    -1, -1, -1, -1, -1, -1,460,440,
    -1, -1, -1, -1, -1, -1, -1,461 }
};


    static readonly byte[] FileToFile = new byte[] { 0, 1, 2, 3, 3, 2, 1, 0 };
    static readonly int[] WdlToMap = new int[] { 1, 3, 0, 2, 0 };
    static readonly byte[] PAFlags = new byte[] { 8, 0, 0, 0, 4 };

    static readonly ulong[,] Binomial = new ulong[7, 64];
    static readonly ulong[,,] PawnIdx = new ulong[2, 6, 24];
    static readonly ulong[,] PawnFactorFile = new ulong[6, 4];
    static readonly ulong[,] PawnFactorRank = new ulong[6, 6];

    static void init_indices()
    {
      int i, j, k;

      // Binomial[k][n] = Bin(n, k)
      for (i = 0; i < 7; i++)
      {
        for (j = 0; j < 64; j++)
        {
          ulong f = 1;
          ulong l = 1;
          for (k = 0; k < i; k++)
          {
            f *= (ulong)(j - k);
            l *= (ulong)(k + 1);
          }
          Binomial[i, j] = f / l;
        }
      }
      for (i = 0; i < 6; i++)
      {
        ulong s = 0;
        for (j = 0; j < 24; j++)
        {
          PawnIdx[0, i, j] = s;
          s += Binomial[i, PawnTwist[0, (1 + (j % 6)) * 8 + (j / 6)]];
          if ((j + 1) % 6 == 0)
          {
            PawnFactorFile[i, j / 6] = s;
            s = 0;
          }
        }
      }

      for (i = 0; i < 6; i++)
      {
        ulong s = 0;
        for (j = 0; j < 24; j++)
        {
          PawnIdx[1, i, j] = s;
          s += Binomial[i, PawnTwist[1, (1 + (j / 4)) * 8 + (j % 4)]];
          if ((j + 1) % 4 == 0)
          {
            PawnFactorRank[i, j / 4] = s;
            s = 0;
          }
        }
      }
    }


    int leading_pawn(Span<int> p, BaseEntry be, int enc)
    {
      for (int i = 1; i < be.pawns0; i++)
      {
        if (Flap[enc - 1, p[0]] > Flap[enc - 1, p[i]])
        {
          Swap(ref p[0], ref p[i]);
        }
      }

      return enc == (int)PieceFileRank.FILE_ENC ? FileToFile[p[0] & 7] : (p[0] - 8) >> 3;
    }

    static ulong encode(Span<int> p, EncInfo ei, BaseEntry be, int enc)
    {
      int n = be.num;
      ulong idx;
      int k;

      if (IntBool(p[0] & 0x04))
      {
        for (int i = 0; i < n; i++)
        {
          p[i] ^= 0x07;
        }
      }

      if (enc == (int)PieceFileRank.PIECE_ENC)
      {
        if (IntBool(p[0] & 0x20))
        {
          for (int i = 0; i < n; i++)
          {
            p[i] ^= 0x38;
          }
        }

        for (int i = 0; i < n; i++)
        {
          if (IntBool(OffDiag[p[i]]))
          {
            if ((OffDiag[p[i]] > 0) && i < (be.kk_enc ? 2 : 3))
              for (int j = 0; j < n; j++)
              {
                p[j] = FlipDiag[p[j]];
              }
            break;
          }
        }

        if (be.kk_enc)
        {
          idx = (ulong)KKIdx[Triangle[p[0]], p[1]];
          k = 2;
        }
        else
        {
          int s1 = BoolInt(p[1] > p[0]);
          int s2 = BoolInt(p[2] > p[0]) + BoolInt(p[2] > p[1]);

          if (IntBool(OffDiag[p[0]]))
          {
            idx = (ulong)(Triangle[p[0]] * 63 * 62 + (p[1] - s1) * 62 + (p[2] - s2));
          }
          else if (IntBool(OffDiag[p[1]]))
          {
            idx = (ulong)(6 * 63 * 62 + Diag[p[0]] * 28 * 62 + Lower[p[1]] * 62 + p[2] - s2);
          }
          else if (IntBool(OffDiag[p[2]]))
          {
            idx = (ulong)(6 * 63 * 62 + 4 * 28 * 62 + Diag[p[0]] * 7 * 28 + (Diag[p[1]] - s1) * 28 + Lower[p[2]]);
          }
          else
          {
            idx = (ulong)(6 * 63 * 62 + 4 * 28 * 62 + 4 * 7 * 28 + Diag[p[0]] * 7 * 6 + (Diag[p[1]] - s1) * 6 + (Diag[p[2]] - s2));
          }
          k = 3;
        }
        idx *= ei.factor[0];
      }
      else
      {
        for (int i = 1; i < be.pawns0; i++)
        {
          for (int j = i + 1; j < be.pawns0; j++)
          {
            if (PawnTwist[enc - 1, p[i]] < PawnTwist[enc - 1, p[j]])
            {
              Swap(ref p[i], ref p[j]);
            }
          }
        }

        k = be.pawns0;
        idx = PawnIdx[enc - 1, k - 1, Flap[enc - 1, p[0]]];
        for (int i = 1; i < k; i++)
        {
          idx += Binomial[k - i, PawnTwist[enc - 1, p[i]]];
        }
        idx *= ei.factor[0];

        // Pawns of other color
        if (be.pawns1 != 0)
        {
          int t = k + be.pawns1;
          for (int i = k; i < t; i++)
          {
            for (int j = i + 1; j < t; j++)
            {
              if (p[i] > p[j])
              {
                Swap(ref p[i], ref p[j]);
              }
            }
          }

          ulong s = 0;
          for (int i = k; i < t; i++)
          {
            int sq = p[i];
            int skips = 0;
            for (int j = 0; j < k; j++)
            {
              skips += BoolInt(sq > p[j]);
            }
            s += Binomial[i - k + 1, sq - skips - 8];
          }

          idx += s * ei.factor[k];

          k = t;
        }
      }

      for (; k < n;)
      {
        int t = k + ei.norm[k];

        for (int i = k; i < t; i++)
        {
          for (int j = i + 1; j < t; j++)
          {
            if (p[i] > p[j]) Swap(ref p[i], ref p[j]);
          }
        }

        ulong s = 0;
        for (int i = k; i < t; i++)
        {
          int sq = p[i];
          int skips = 0;
          for (int j = 0; j < k; j++)
          {
            skips += (sq > p[j]) ? 1 : 0;
          }
          s += Binomial[i - k + 1, sq - skips];
        }

        idx += s * ei.factor[k];
        k = t;
      }

      return idx;
    }





    bool test_tb(string str, string suffix)
    {
      string fn = get_fn(str, suffix);

      if (!File.Exists(fn))
      {
        if (VERBOSE) Console.WriteLine("Tablebase file missing " + str + suffix); // TODO: cleanup
        return false;
      }
      else
      {
        if (VERBOSE) Console.WriteLine("load " + fn);
      }

      FileInfo fileInfo = new FileInfo(fn);
      if ((fileInfo.Length & 63) != 16)
      {
        throw new Exception($"Incomplete tablebase file  {fn}");
        return false;
        //printf("info string Incomplete tablebase file %s.%s\n", str, suffix); 
      }

      return true;
    }

    void init_tb(string str)
    {
      if (!test_tb(str, tbSuffix[(int)WDL.WDL]))
      {
        return;
      }


      int[] pcs = new int[16];
      int color = 0;

      foreach (char s in str)
      {
        if (s == 'v')
        {
          color = 8;
        }
        else
        {
          FathomPieceType piece_type = char_to_piece_type(s);
          if (piece_type > 0)
          {
            //Debug.Assert((piece_type | color) < 16);
            pcs[(int)piece_type | color]++;
          }
        }
      }

      ulong key = calc_key_from_pcs(pcs, 0);
      ulong key2 = calc_key_from_pcs(pcs, 1);

      bool hasPawns = pcs[(int)FathomPiece.W_PAWN] > 0 || pcs[(int)FathomPiece.B_PAWN] > 0;
      BaseEntry be = hasPawns ? pawnEntry[tbNumPawn++]
                              : pieceEntry[tbNumPiece++];



      be.hasPawns = hasPawns;
      be.key = key;
      be.symmetric = key == key2;
      be.num = 0;
      for (int i = 0; i < 16; i++)
      {
        be.num += (byte)pcs[i];
      }

      numWdl++;
      numDtm += (be.hasDtm = test_tb(str, tbSuffix[(int)WDL.DTM])) ? 1 : 0;
      numDtz += (be.hasDtz = test_tb(str, tbSuffix[(int)WDL.DTZ])) ? 1 : 0;

      if (be.num > TB_MaxCardinality)
      {
        TB_MaxCardinality = be.num;
      }
      if (be.hasDtm)
      {
        if (be.num > TB_MaxCardinalityDTM)
        {
          TB_MaxCardinalityDTM = be.num;
        }
      }

      if (!be.hasPawns)
      {
        int j = 0;
        for (int i = 0; i < 16; i++)
        {
          if (pcs[i] == 1)
          {
            j++;
          }
        }
        be.kk_enc = j == 2;
      }
      else
      {
        be.pawns0 = (byte)pcs[(int)FathomPiece.W_PAWN];
        be.pawns1 = (byte)pcs[(int)FathomPiece.B_PAWN];
        if (pcs[(int)FathomPiece.B_PAWN] != 0 && (pcs[(int)FathomPiece.W_PAWN] == 0 || pcs[(int)FathomPiece.W_PAWN] > pcs[(int)FathomPiece.B_PAWN]))
        {
          Swap(ref be.pawns0, ref be.pawns1);
        }
      }

      add_to_hash(be, key);
      if (key != key2)
      {
        add_to_hash(be, key2);
      }

    }

    void add_to_hash(BaseEntry ptr, ulong key)
    {
      int idx;

      idx = (int)(key >> (64 - TB_HASHBITS));
      while (tbHash[idx].ptr != null)
      {
        idx = (idx + 1) & ((1 << TB_HASHBITS) - 1);
      }

      tbHash[idx].key = key;
      tbHash[idx].ptr = ptr;
    }


    static void Swap<T>(ref T a, ref T b)
    {
      T temp = a;
      a = b;
      b = temp;
    }

    bool staticsInitialized = false;

    // map upper-case characters to piece types
    static FathomPieceType char_to_piece_type(char c)
    {
      for (FathomPieceType pt = FathomPieceType.PAWN; pt <= FathomPieceType.KING; pt++)
      {
        if (c == piece_to_char[(int)pt])
        {
          return pt;
        }
      }
      return (FathomPieceType)0;
    }


    static string piece_to_char = " PNBRQK  pnbrqk";

    static char pchr(int i) => piece_to_char[(int)FathomPieceType.QUEEN - (i)];


    internal unsafe struct TbHashEntry
    {
      public ulong key;
      public BaseEntry ptr;
    };


    static int tbNumPiece, tbNumPawn;
    internal static int numWdl, numDtm, numDtz;

    PieceEntry[] pieceEntry;
    PawnEntry[] pawnEntry;
    TbHashEntry[] tbHash = new TbHashEntry[1 << TB_HASHBITS];

    bool dtmLossOnly;



    #region

    const int TB_MAX_MOVES = (192 + 1);
    const int TB_MAX_CAPTURES = 64;
    const int TB_MAX_PLY = 256;
    const int TB_CASTLING_K = 0x1;     /* White king-side. */
    const int TB_CASTLING_Q = 0x2;     /* White queen-side. */
    const int TB_CASTLING_k = 0x4;     /* Black king-side. */
    const int TB_CASTLING_q = 0x8;     /* Black queen-side. */

    const int TB_PROMOTES_NONE = 0;
    const int TB_PROMOTES_QUEEN = 1;
    const int TB_PROMOTES_ROOK = 2;
    const int TB_PROMOTES_BISHOP = 3;
    const int TB_PROMOTES_KNIGHT = 4;

    const uint TB_RESULT_WDL_MASK = 0x0000000F;
    const uint TB_RESULT_TO_MASK = 0x000003F0;
    const uint TB_RESULT_FROM_MASK = 0x0000FC00;
    const uint TB_RESULT_PROMOTES_MASK = 0x00070000;
    const uint TB_RESULT_EP_MASK = 0x00080000;
    const uint TB_RESULT_DTZ_MASK = 0xFFF00000;
    const int TB_RESULT_WDL_SHIFT = 0;
    const int TB_RESULT_TO_SHIFT = 4;
    const int TB_RESULT_FROM_SHIFT = 10;
    const int TB_RESULT_PROMOTES_SHIFT = 16;
    const int TB_RESULT_EP_SHIFT = 19;
    const int TB_RESULT_DTZ_SHIFT = 20;


    static uint TB_SET_WDL(uint _res, int _wdl) => (uint)(((_res) & ~TB_RESULT_WDL_MASK) | (((_wdl) << TB_RESULT_WDL_SHIFT) & TB_RESULT_WDL_MASK));

    internal static uint TB_GET_WDL(int _res) => (uint)((_res) & TB_RESULT_WDL_MASK) >> TB_RESULT_WDL_SHIFT;

    internal static uint TB_GET_TO(int _res) => (uint)((_res) & TB_RESULT_TO_MASK) >> TB_RESULT_TO_SHIFT;
    internal static uint TB_GET_FROM(int _res) => (uint)((_res) & TB_RESULT_FROM_MASK) >> TB_RESULT_FROM_SHIFT;
    internal static uint TB_GET_PROMOTES(int _res) => (uint)((_res) & TB_RESULT_PROMOTES_MASK) >> TB_RESULT_PROMOTES_SHIFT;
    internal static uint TB_GET_EP(int _res) => (uint)((_res) & TB_RESULT_EP_MASK) >> TB_RESULT_EP_SHIFT;
    internal static uint TB_GET_DTZ(int _res) => (uint) ((_res) & TB_RESULT_DTZ_MASK) >> TB_RESULT_DTZ_SHIFT;

    static uint TB_SET_DTZ(uint _res, uint _dtz) => (uint)((_res) & ~TB_RESULT_DTZ_MASK) | (((_dtz) << TB_RESULT_DTZ_SHIFT) & TB_RESULT_DTZ_MASK);
    static uint TB_SET_TO(uint _res, uint _to) => (uint)((_res) & ~TB_RESULT_TO_MASK) | (((_to) << TB_RESULT_TO_SHIFT) & TB_RESULT_TO_MASK);
    static uint TB_SET_FROM(uint _res, uint _from) => (uint)((_res) & ~TB_RESULT_FROM_MASK) | (((_from) << TB_RESULT_FROM_SHIFT) & TB_RESULT_FROM_MASK);

    static uint TB_SET_PROMOTES(uint _res, uint _promotes) => (uint)(((_res) & ~TB_RESULT_PROMOTES_MASK) | (((_promotes) << TB_RESULT_PROMOTES_SHIFT) & TB_RESULT_PROMOTES_MASK));
    static uint TB_SET_EP(uint _res, uint _ep) => (uint)(((_res) & ~TB_RESULT_EP_MASK) | (((_ep) << TB_RESULT_EP_SHIFT) & TB_RESULT_EP_MASK));


    internal const uint TB_RESULT_FAILED = 0xFFFFFFFF;
    internal static readonly uint TB_RESULT_CHECKMATE = TB_SET_WDL(0, (int)FathomWDLResult.Win); // TODO: make this truly const
    internal static readonly uint TB_RESULT_STALEMATE = TB_SET_WDL(0, (int)FathomWDLResult.Draw); // TODO: make this truly cons


    uint tb_probe_wdl(ulong white, ulong black, ulong kings, ulong queens, ulong rooks, ulong bishops, ulong knights, ulong pawns, uint ep, bool turn)
    {
      FathomPos pos = new FathomPos(white, black, kings, queens, rooks, bishops, knights, pawns, 0, (byte)ep, turn, 0, 0);
      int success;
      int v = probe_wdl(in pos, out success);
      if (success == 0)
      {
        return TB_RESULT_FAILED;
      }
      return (uint)(v + 2);
    }

    static uint dtz_to_wdl(uint cnt50, int dtz)
    {
      uint wdl = 0;
      if (dtz > 0)
      {
        wdl = (uint)(dtz + cnt50 <= 100 ? 2 : 1);
      }
      else if (dtz < 0)
      {
        wdl = (uint)(-dtz + cnt50 <= 100 ? -2 : -1);
      }
      return wdl + 2;
    }

    public uint tb_probe_root(ulong white, ulong black, ulong kings, ulong queens, ulong rooks, ulong bishops,
                              ulong knights, ulong pawns, uint rule50, int castling, ulong ep, bool turn, uint[] results)
    {
      if (castling != 0)
      {
        return TB_RESULT_FAILED;
      }


      FathomPos pos = new FathomPos(white, black, kings, queens, rooks, bishops, knights, pawns, (byte)rule50, (byte)ep, turn, 0, 0);


      if (!FathomMoveGen.is_valid(in pos))
      {
        return TB_RESULT_FAILED;
      }

      int dtz;
      ushort move = probe_root(in pos, out dtz, results);


      if (move == 0)
      {
        return TB_RESULT_FAILED;
      }
      if (move == MOVE_CHECKMATE)
      {
        return TB_RESULT_CHECKMATE;
      }
      if (move == MOVE_STALEMATE)
      {
        return TB_RESULT_STALEMATE;
      }

      uint res = 0;

      res = TB_SET_WDL(res, (int)dtz_to_wdl(rule50, dtz));
      res = TB_SET_DTZ(res, (uint)(dtz < 0 ? -dtz : dtz));
      res = TB_SET_FROM(res, move_from(move));
      res = TB_SET_TO(res, move_to(move));
      res = TB_SET_PROMOTES(res, move_promotes(move));
      res = TB_SET_EP(res, is_en_passant(in pos, move) ? (uint)1 : 0);
      return res;
    }

    int tb_probe_root_dtz(ulong white, ulong black, ulong kings, ulong queens, ulong rooks, ulong bishops, ulong knights, ulong pawns,
                          ulong rule50, ulong castling, ulong ep, bool turn, bool hasRepeated, bool useRule50, List<TbRootMove> results)
    {
      FathomPos pos = new(white, black, kings, queens, rooks, bishops, knights, pawns, (byte)rule50, (byte)ep, turn, 0, 0);

      if (castling != 0)
      {
        return 0;
      }


      return root_probe_dtz(in pos, hasRepeated, useRule50, results);
    }

    int tb_probe_root_wdl(ulong white, ulong black, ulong kings, ulong queens, ulong rooks, ulong bishops, ulong knights, ulong pawns,
                          int rule50, int castling, int ep, bool turn, bool useRule50, List<TbRootMove> results)
    {
      FathomPos pos = new(white, black, kings, queens, rooks, bishops, knights, pawns, (byte)rule50, (byte)ep, turn, 0, 0);
      if (castling != 0)
      {
        results = default;
        return 0;
      }

      return root_probe_wdl(in pos, useRule50, results);
    }

    // Given a position, produce a text string of the form KQPvKRP, where
    // "KQP" represents the white pieces if flip == false and the black pieces
    // if flip == true.
    static string prt_str(in FathomPos pos, bool flip)
    {
      int color = (int)(flip ? FathomColor.BLACK : FathomColor.WHITE);

      StringBuilder sb = new();

      for (int pt = (int)FathomPieceType.KING; pt >= (int)FathomPieceType.PAWN; pt--)
      {
        for (int i = popcount(pieces_by_type(pos, (FathomColor)color, (FathomPieceType)pt)); i > 0; i--)
        {
          sb.Append(piece_to_char[pt]);
        }
      }

      sb.Append("v");

      color ^= 1;

      for (int pt = (int)FathomPieceType.KING; pt >= (int)FathomPieceType.PAWN; pt--)
      {
        for (int i = popcount(pieces_by_type(pos, (FathomColor)color, (FathomPieceType)pt)); i > 0; i--)
        {
          sb.Append(piece_to_char[pt]);
        }
      }

      return sb.ToString();
    }

    #endregion

    unsafe string get_fn(string filename, string suffix)
    {
      foreach (string path in paths.Split(SEP_CHAR))
      {
        string fn = Path.Combine(path, filename) + suffix;
        if (File.Exists(fn))
        {
          return fn;
        }
      }
      return null;
    }

    unsafe void* open_tb(string filename, string suffix)
    {
      string fn = get_fn(filename, suffix);

      MemoryMappedFile mmf;
      MemoryMappedViewAccessor mmView;

      // TODO: release these files somewhere!
      //mmView.Dispose();
      //mmf.Dispose();


      mmf = MemoryMappedFile.CreateFromFile(fn, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);

      mmView = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);


      // TODO: more error checking
      //        fprintf(stderr, "Could not map %s%s into memory.\n", name, suffix);
      //        exit(EXIT_FAILURE);

      unsafe
      {
        byte* ptr = default;
        mmView.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
        return ptr;
      }

      return null;
    }


    static int num_tables(BaseEntry be, int type)
    {
      return be.hasPawns ? type == (int)WDL.DTM ? 6 : 4 : 1;
    }



    unsafe bool init_table(BaseEntry be, string tbFileName, int type)
    {
      string suffix = tbSuffix[type];


      byte* data = (byte*)open_tb(tbFileName, suffix);

      if (data == null)
      {
        return false;
      }

      if (read_le_u32(data) != tbMagic[type])
      {
        Console.WriteLine($"Corrupted table {tbFileName}.{suffix}");
        //unmap_file((void*) data, be->mapping[type]);
        return false;
      }

      be.data[type] = data;

      bool split = type != (int)WDL.DTZ && (IntBool(data[4] & 0x01));
      if (type == (int)WDL.DTM)
      {
        be.dtmLossOnly = IntBool(data[4] & 0x04);
      }

      data += 5;

      ulong[,] tb_size = new ulong[6, 2];// TODO: make stackalloc?
      int num = num_tables(be, type);
      Span<EncInfo> ei = be.first_ei(type);
      int enc = !be.hasPawns ? (int)PieceFileRank.PIECE_ENC : type != (int)WDL.DTM ? (int)PieceFileRank.FILE_ENC : (int)PieceFileRank.RANK_ENC;

      for (int t = 0; t < num; t++)
      {
        tb_size[t, 0] = init_enc_info(ref ei[t], be, data, 0, t, enc);
        if (split)
        {
          tb_size[t, 1] = init_enc_info(ref ei[num + t], be, data, 4, t, enc);
        }
        data += be.num + 1 + BoolInt((be.hasPawns && IntBool(be.pawns1)));
      }

      data += (uint)data & 1;

      ulong[,][] size = new ulong[6, 2][];
      for (int i = 0; i < 6; i++)
      {
        for (int j = 0; j < 2; j++)
        {
          size[i, j] = new ulong[3]; // TODO: make stackalloc:
        }
      }

      for (int t = 0; t < num; t++)
      {
        byte flags = default;

        ei[t].precomp = setup_pairs(ref data, tb_size[t, 0], size[t, 0], &flags, type);
        if (type == (int)(WDL.DTZ))
        {
          if (!be.hasPawns)
          {
            (be as PieceEntry).dtzFlags[0] = flags;
          }
          else
          {
            (be as PawnEntry).dtzFlags[t] = flags;
          }
        }
        if (split)
        {
          ei[num + t].precomp = setup_pairs(ref data, tb_size[t, 1], size[t, 1], &flags, type);
        }
        else if (type != (int)WDL.DTZ)
        {
          ei[num + t].precomp = default;
        }
      }


      if (type == (int)WDL.DTM && !be.dtmLossOnly)
      {
        ushort* map = (ushort*)data;
        if (be is PieceEntry)
        {
          (be as PieceEntry).dtmMap = map;
        }
        else
        {
          (be as PawnEntry).dtmMap = map;
        }


        ref ushort[,,] refMapIdx = ref (be is PawnEntry) ? ref ((be as PawnEntry).dtmMapIdx)
                                                        : ref ((be as PieceEntry).dtmMapIdx);

        //  ushort(*mapIdx)[2][2] = be->hasPawns ? &PAWN(be)->dtmMapIdx[0] : &PIECE(be)->dtmMapIdx;
        for (int t = 0; t < num; t++)
        {
          for (int i = 0; i < 2; i++)
          {
            refMapIdx[t, 0, i] = (ushort)(data + 1 - (byte*)map);
            data += 2 + 2 * read_le_u16(data);
          }
          if (split)
          {
            for (int i = 0; i < 2; i++)
            {
              refMapIdx[t, 1, i] = (ushort)(data + 1 - (byte*)map);
              data += 2 + 2 * read_le_u16(data);
            }
          }
        }
      }

      if (type == (int)WDL.DTZ)
      {
        //  void* map = data;
        //  *(be->hasPawns ? &PAWN(be)->dtzMap : &PIECE(be)->dtzMap) = map;

        ushort* map1 = (ushort*)data;
        if (be is PieceEntry)
        {
          (be as PieceEntry).dtzMap = map1;
        }
        else
        {
          (be as PawnEntry).dtzMap = map1;
        }

        //      ushort(*mapIdx)[4] = be->hasPawns ? &PAWN(be)->dtzMapIdx[0]
        //                                        : &PIECE(be)->dtzMapIdx;
        ref ushort[,] refMapIdx = ref (be is PawnEntry) ? ref ((be as PawnEntry).dtzMapIdx)
                                                        : ref ((be as PieceEntry).dtzMapIdx);

        byte[] flags = be.hasPawns ? (be as PawnEntry).dtzFlags
                                      : (be as PieceEntry).dtzFlags;
        for (int t = 0; t < num; t++)
        {
          if (IntBool(flags[t] & 2))
          {
            if (!IntBool((flags[t] & 16)))
            {
              for (int i = 0; i < 4; i++)
              {
                refMapIdx[t, i] = (ushort)(data + 1 - (byte*)map1);
                data += 1 + data[0];
              }
            }
            else
            {
              data += ((IntPtr)data).ToInt64() & 0x01;
              for (int i = 0; i < 4; i++)
              {
                refMapIdx[t, i] = (ushort)(data + 1 - (byte*)map1);
                data += 2 + 2 * read_le_u16(data);
              }
            }
          }
        }
        data += ((IntPtr)data).ToInt64() & 0x01;
      }


      for (int t = 0; t < num; t++)
      {
        ei[t].precomp.Ref.indexTable = data;
        data += size[t, 0][0];
        if (split)
        {
          ei[num + t].precomp.Ref.indexTable = data;
          data += size[t, 1][0];
        }
      }

      for (int t = 0; t < num; t++)
      {
        ei[t].precomp.Ref.sizeTable = (ushort*)data;
        if (*ei[t].precomp.Ref.sizeTable < 0)
        {
          throw new Exception("Internal error: Unexpected negative sized table entry found in tablebase");
        }
        data += size[t, 0][1];
        if (split)
        {
          ei[num + t].precomp.Ref.sizeTable = (ushort*)data;
          data += size[t, 1][1];
        }
      }

      for (int t = 0; t < num; t++)
      {
        data = (byte*)((((IntPtr)data).ToInt64() + 0x3f) & ~0x3f);
        ei[t].precomp.Ref.data = data;
        data += size[t, 0][2];
        if (split)
        {
          data = (byte*)((((IntPtr)data).ToInt64() + 0x3f) & ~0x3f);
          ei[num + t].precomp.Ref.data = data;
          data += size[t, 1][2];
        }
      }

      if (type == (int)WDL.DTM && be.hasPawns)
      {
        (be as PawnEntry).dtmSwitched = calc_key_from_pieces(ei[0].pieces, be.num) != be.key;
      }

      return true;
    }


    List<PairsDataPtr> allocatedPairsData = new();


    unsafe PairsDataPtr setup_pairs(ref byte* ptr, ulong tb_size, ulong[] size, byte* flags, int type)
    {
      byte* data = ptr;

      *flags = data[0];
      if (IntBool(data[0] & 0x80))
      {
        PairsDataPtr dEmpty = new PairsDataPtr(0);
        allocatedPairsData.Add(dEmpty);
        ref PairsData refPDEmpty = ref dEmpty.Ref;
        refPDEmpty.idxBits = 0;
        refPDEmpty.constValue[0] = type == (int)WDL.WDL ? data[1] : (byte)0;
        refPDEmpty.constValue[1] = 0;
        ptr = data + 2;
        size[0] = size[1] = size[2] = 0;
        return dEmpty;
      }

      byte blockSize = data[1];
      byte idxBits = data[2];
      uint realNumBlocks = read_le_u32(data + 4);
      uint numBlocks = realNumBlocks + data[3];
      byte maxLen = data[8];
      byte minLen = data[9];
      int h = maxLen - minLen + 1;
      int numSyms = (int)read_le_u16(data + 10 + 2 * h);

      int extraBytes = h * (int)sizeof(ulong) + numSyms;

      // Add 8 more bytes because the original C structure had an additional field (uint64_t base[1])
      extraBytes += sizeof(ulong);

      PairsDataPtr d = new PairsDataPtr(extraBytes);
      allocatedPairsData.Add(d);
      ref PairsData refPD = ref d.Ref;
      //    d = (struct PairsData*)malloc(sizeof(struct PairsData) +h * sizeof(uint64_t) + numSyms);

      refPD.blockSize = blockSize;
      refPD.idxBits = idxBits;
      refPD.offset = (ushort*)(&data[10]);
      IntPtr symLen = IntPtr.Add(d.rawData, sizeof(PairsData) + h * sizeof(ulong));
      refPD.symLen = (byte*)symLen.ToPointer();
      //refPD.symLen = (byte*)d + sizeof(struct PairsData) +h* sizeof(uint64_t);

      refPD.symPat = &data[12 + 2 * h];
      refPD.minLen = minLen;

      int ptrIncrement = 12 + 2 * h + 3 * numSyms + (numSyms & 1);
      ptr = ptr + ptrIncrement;

      ulong num_indices = (tb_size + (ulong)(1 << idxBits) - 1) >> idxBits;
      size[0] = 6 * num_indices;
      size[1] = (2 * numBlocks);
      size[2] = ((ulong)realNumBlocks << blockSize);

      Debug.Assert(numSyms < TB_MAX_SYMS);

      Span<byte> tmp = stackalloc byte[TB_MAX_SYMS];
      for (int s = 0; s < numSyms; s++)
      {
        if (tmp[s] == 0)
        {
          calc_symLen(ref refPD, (uint)s, tmp);
        }
      }

      d.baseData[h - 1] = 0;


      for (int i = h - 2; i >= 0; i--)
      {
        d.baseData[i] = ((d.baseData[i + 1] + read_le_u16((byte*)(refPD.offset + i)) - read_le_u16((byte*)(refPD.offset + i + 1))) / 2);
      }

      for (int i = 0; i < h; i++)
      {
        d.baseData[i] <<= 64 - (minLen + i);
      }

      refPD.offset -= refPD.minLen;

      return d;
    }


    static unsafe ushort read_le_u16(void* p)
    {
      return from_le_u16(*(ushort*)p);
    }

    static unsafe void calc_symLen(ref PairsData d, uint s, Span<byte> tmp)
    {
      byte* w = d.symPat + 3 * s;
      int s2Int = (w[2] << 4) | (w[1] >> 4);
      uint s2 = (uint)s2Int;
      if (s2 == 0x0fff)
      {
        d.symLen[s] = 0;
      }
      else
      {
        int s1Int = ((w[1] & 0xf) << 8) | w[0];
        uint s1 = (uint)s1Int;
        if (tmp[(int)s1] == 0) calc_symLen(ref d, s1, tmp);
        if (tmp[(int)s2] == 0) calc_symLen(ref d, s2, tmp);
        d.symLen[s] = (byte)(d.symLen[s1] + d.symLen[s2] + 1);
      }

      tmp[(int)s] = 1;
    }


    #region Root moves

    struct TbRootMove
    {
      public ushort move;
      public ushort[] pv; //[TB_MAX_PLY] 
      public int pvSize;
      public int tbScore;
      public int tbRank;
    };


    #endregion

    #region Helpers


    static unsafe ulong encode_piece(Span<int> p, EncInfo ei, BaseEntry be)
    {
      return encode(p, ei, be, (int)PieceFileRank.PIECE_ENC);
    }

    static unsafe ulong encode_pawn_f(Span<int> p, EncInfo ei, BaseEntry be)
    {
      return encode(p, ei, be, (int)PieceFileRank.FILE_ENC);
    }

    static unsafe ulong encode_pawn_r(Span<int> p, EncInfo ei, BaseEntry be)
    {
      return encode(p, ei, be, (int)PieceFileRank.RANK_ENC);
    }


    unsafe static ulong init_enc_info(ref EncInfo ei, BaseEntry be, byte* tb, int shift, int t, int enc)
    {
      bool morePawns = enc != (int)PieceFileRank.PIECE_ENC && be.pawns1 > 0;

      for (int i = 0; i < be.num; i++)
      {
        ei.pieces[i] = (byte)((tb[i + 1 + BoolInt(morePawns)] >> shift) & 0x0f);
        ei.norm[i] = 0;
      }

      int order = (tb[0] >> shift) & 0x0f;
      int order2 = morePawns ? (tb[1] >> shift) & 0x0f : 0x0f;

      int k = ei.norm[0] = (byte)(enc != (int)PieceFileRank.PIECE_ENC ? be.pawns0 : be.kk_enc ? 2 : 3);

      if (morePawns)
      {
        ei.norm[k] = be.pawns1;
        k += ei.norm[k];
      }

      for (int i = k; i < be.num; i += ei.norm[i])
      {
        for (int j = i; j < be.num && ei.pieces[j] == ei.pieces[i]; j++)
        {
          ei.norm[i]++;
        }
      }

      int n = 64 - k;
      ulong f = 1;

      for (int i = 0; k < be.num || i == order || i == order2; i++)
      {
        if (i == order)
        {
          ei.factor[0] = f;
          f *= enc == (int)PieceFileRank.FILE_ENC ? PawnFactorFile[ei.norm[0] - 1, t]
              : enc == (int)PieceFileRank.RANK_ENC ? PawnFactorRank[ei.norm[0] - 1, t]
                                                   : be.kk_enc ? (ulong)462 : (ulong)31332;
        }
        else if (i == order2)
        {
          ei.factor[ei.norm[0]] = f;
          f *= (ulong)subfactor(ei.norm[ei.norm[0]], (ulong)(48 - ei.norm[0]));
        }
        else
        {
          ei.factor[k] = f;
          f *= (ulong)subfactor(ei.norm[k], (ulong)n);
          n -= ei.norm[k];
          k += ei.norm[k];
        }
      }

      return f;
    }


    unsafe Span<byte> decompress_pairs(PairsDataPtr dPtr, ulong idx)
    {
      ref PairsData d = ref dPtr.Ref;
      if (!IntBool(d.idxBits))
      {
        return d.constValueSpan;
      }

      uint mainIdx = (uint)(idx >> d.idxBits);
      int litIdx = (int)((int)idx & (((int)1 << d.idxBits) - 1)) - ((int)1 << (d.idxBits - 1)); // DJE

      uint block;
      uint* blockPtr = (uint*)(((IntPtr)d.indexTable).ToInt64() + 6 * mainIdx);
      block = *blockPtr;
      //  memcpy(&block, d->indexTable + 6 * mainIdx, sizeof(block));
      block = from_le_u32(block);


      ushort idxOffset = *(ushort*)(((IntPtr)d.indexTable).ToInt64() + 6 * mainIdx + 4);
      litIdx += from_le_u16(idxOffset);

      if (litIdx < 0)
      {
        while (litIdx < 0)
          litIdx += d.sizeTable[--block] + 1;
      }
      else
      {
        while (litIdx > d.sizeTable[block])
          litIdx -= d.sizeTable[block++] + 1;
      }


      uint* ptr = (uint*)(d.data + ((ulong)block << d.blockSize));

      int m = d.minLen;
      ushort* offset = d.offset;

      ulong* basePtr = dPtr.baseData - m;// * sizeof(IntPtr);
      byte* symLen = d.symLen;
      uint sym;
      uint bitCnt = 0;// number of "empty bits" in code

      ulong code = from_be_u64(*(ulong*)ptr);

      ptr += 2;

      for (; ; )
      {
        int l = m;
        while (code < basePtr[l])
        {
          l++;
        }
        sym = from_le_u16(offset[l]);
        sym += (uint)((code - basePtr[l]) >> (64 - l));
        if (litIdx < (int)symLen[sym] + 1) break;
        litIdx -= (int)symLen[sym] + 1;
        code <<= l;
        bitCnt += (uint)l;
        if (bitCnt >= 32)
        {
          bitCnt -= 32;
          uint tmp = from_be_u32(*ptr++);
          code |= (ulong)tmp << (int)bitCnt;
        }
      }


      byte* symPat = d.symPat;
      while (symLen[sym] != 0)
      {
        byte* w = symPat + (3 * sym);
        int s1 = ((w[1] & 0xf) << 8) | w[0];
        if (litIdx < (int)symLen[s1] + 1)
        {
          sym = (uint)s1;
        }
        else
        {
          litIdx -= (int)symLen[s1] + 1;
          sym = (uint)((w[2] << 4) | (w[1] >> 4));
        }
      }

      byte* retSymPat = symPat + 3 * sym;
      return new Span<byte>(retSymPat, 2);
      //return &symPat[3 * sym];

    }



    // p[i] is to contain the square 0-63 (A1-H8) for a piece of type
    // pc[i] ^ flip, where 1 = white pawn, ..., 14 = black king and pc ^ flip
    // flips between white and black if flip == true.
    // Pieces of the same type are guaranteed to be consecutive.
    unsafe int fill_squares(in FathomPos pos, Span<byte> pc, bool flip, int mirror, Span<int> p, int i)
    {
      FathomColor color = ColorOfPiece(pc[i]);
      if (flip)
      {
        color = color == FathomColor.WHITE ? FathomColor.BLACK : FathomColor.WHITE;
      }
      ulong bb = pieces_by_type(pos, color, TypeOfPiece(pc[i]));
      int sq;
      do
      {
        sq = lsb(bb);
        p[i++] = (sq ^ mirror);
        bb = poplsb(bb);
      } while (bb != 0);
      return i;
    }


    static bool IntBool(int i) => i != 0 ? true : false;
    static int BoolInt(bool b) => b ? 1 : 0;

    unsafe int probe_table(in FathomPos pos, int s, ref int success, int type)
    {
      // Obtain the position's material-signature key
      ulong key = calc_key(in pos, false);

      // Test for KvK
      // Note: Cfish has key == 2ULL for KvK but we have 0
      if (type == (int)WDL.WDL && key == 0)
      {
        return 0;
      }

      int hashIdx = (int)(key >> (64 - TB_HASHBITS));
      while (tbHash[hashIdx].key != 0 && tbHash[hashIdx].key != key)
      {
        hashIdx = (hashIdx + 1) & ((1 << TB_HASHBITS) - 1);
      }

      if (tbHash[hashIdx].ptr == null)
      {
        success = 0;
        return 0;
      }

      BaseEntry be = tbHash[hashIdx].ptr;
      if ((type == (int)WDL.DTM && !be.hasDtm) || (type == (int)WDL.DTZ && !be.hasDtz))
      {
        success = 0;
        return 0;
      }

      if (!be.initialized[type])
      {
        string str = prt_str(in pos, be.key != key);

        lock (be.readyLockObj[type])
        {
          if (!be.initialized[type])
          {
            if (!init_table(be, str, type))
            {
              tbHash[hashIdx].ptr = null; // mark as deleted
              success = 0;
              return 0;
            }
            else
            {
              be.initialized[type] = true;
            }
          }
        }
      }

      bool bside, flip;
      if (!be.symmetric)
      {
        flip = key != be.key;
        bside = (BoolInt(pos.turn) == (int)FathomColor.WHITE) == flip;
        if (type == (int)WDL.DTM && be.hasPawns && (be as PawnEntry).dtmSwitched)
        {
          flip = !flip;
          bside = !bside;
        }
      }
      else
      {
        flip = pos.turn != IntBool((int)FathomColor.WHITE);
        bside = false;
      }

      Span<EncInfo> ei = be.first_ei(type);
      ref EncInfo thisEI = ref ei[0];
      Span<int> p = stackalloc int[TB_PIECES];
      ulong idx;
      int t = 0;
      byte flags = 0;

      if (!be.hasPawns)
      {
        if (type == (int)WDL.DTZ)
        {
          flags = (be as PieceEntry).dtzFlags[0];
          if (IntBool(flags & 1) != bside && !be.symmetric)
          {
            success = -1;
            return 0;
          }
        }

        thisEI = ref type != (int)WDL.DTZ ? ref ei[BoolInt(bside)] : ref ei[0];
        for (int i = 0; i < be.num;)
        {
          i = fill_squares(pos, thisEI.pieces, flip, 0, p, i);
        }

        idx = encode_piece(p, thisEI, be);
      }
      else
      {
        int i = fill_squares(pos, ei[0].pieces, flip, flip ? 0x38 : 0, p, 0);
        t = leading_pawn(p, be, type != (int)WDL.DTM ? (int)PieceFileRank.FILE_ENC : (int)PieceFileRank.RANK_ENC);
        if (type == (int)WDL.DTZ)
        {
          flags = (be as PawnEntry).dtzFlags[t];
          if (((flags & 1) == 0 ? false : true) != bside && !be.symmetric)
          {
            success = -1;
            return 0;
          }
        }

        thisEI = ref type == (int)WDL.WDL ? ref ei[t + 4 * BoolInt(bside)]
                                          : ref type == (int)WDL.DTM ? ref ei[t + 6 * BoolInt(bside)] : ref ei[t];
        while (i < be.num)
        {
          i = fill_squares(pos, thisEI.pieces, flip, flip ? 0x38 : 0, p, i);
        }
        idx = type != (int)WDL.DTM ? encode_pawn_f(p, thisEI, be) : encode_pawn_r(p, thisEI, be);
      }


      Span<byte> w = decompress_pairs(thisEI.precomp, idx);

      if (type == (int)WDL.WDL)
      {
        return (int)w[0] - 2;
      }

      int v = w[0] + ((w[1] & 0x0f) << 8);

      if (type == (int)WDL.DTM)
      {
        if (!be.dtmLossOnly)
        {
          v = (int)from_le_u16(be.hasPawns
                           ? (be as PawnEntry).dtmMap[(be as PawnEntry).dtmMapIdx[t, bside ? 1 : 0, s] + v]
                            : (be as PieceEntry).dtmMap[(be as PieceEntry).dtmMapIdx[0, bside ? 1 : 0, s] + v]);
        }
      }
      else
      {
        if ((flags & 2) != 0)
        {
          int m = WdlToMap[s + 2];
          if ((flags & 16) == 0)
          {
            v = be.hasPawns
               ? ((byte*)(be as PawnEntry).dtzMap)[(be as PawnEntry).dtzMapIdx[t, m] + v]
               : ((byte*)(be as PieceEntry).dtzMap)[(be as PieceEntry).dtzMapIdx[0, m] + v];
          }
          else
          {
            v = (int)from_le_u16(be.hasPawns
                             ? ((byte*)(be as PawnEntry).dtzMap)[(be as PawnEntry).dtzMapIdx[t, m] + v]
                              : ((byte*)(be as PieceEntry).dtzMap)[(be as PieceEntry).dtzMapIdx[0, m] + v]);
          }
        }

        if ((flags & PAFlags[s + 2]) == 0 || (s & 1) != 0)
        {
          v *= 2;
        }
      }

      return v;
    }

    int probe_wdl_table(in FathomPos pos, ref int success)
    {
      return probe_table(pos, 0, ref success, (int)WDL.WDL);
    }

    int probe_dtm_table(in FathomPos pos, int won, ref int success)
    {
      return probe_table(pos, won, ref success, (int)WDL.DTM);
    }

    int probe_dtz_table(in FathomPos pos, int wdl, ref int success)
    {
      return probe_table(pos, wdl, ref success, (int)WDL.DTZ);
    }


    // probe_ab() is not called for positions with en passant captures.
    int probe_ab(in FathomPos pos, int alpha, int beta, ref int success)
    {
      Debug.Assert(pos.ep == 0);


      // Generate (at least) all legal captures including (under)promotions.
      // It is OK to generate more, as long as they are filtered out below.
      TBMoveList moves = gen_captures(in pos);
      for (int i = 0; i < moves.NumMoves; i++)
      {
        FathomPos pos1 = default;
        ushort move = moves.Moves[i];
        if (!is_capture(pos, move))
        {
          continue;
        }

        if (!do_move(ref pos1, in pos, move))
        {
          continue; // illegal move
        }

        int vr = -probe_ab(in pos1, -beta, -alpha, ref success);

        if (success == 0) return 0;

        if (vr > alpha)
        {
          if (vr >= beta)
          {
            return vr;
          }
          alpha = vr;
        }
      }

      int vTable = probe_wdl_table(pos, ref success);

      return alpha >= vTable ? alpha : vTable;
    }


    // Probe the WDL table for a particular position.
    //
    // If *success != 0, the probe was successful.
    //
    // If *success == 2, the position has a winning capture, or the position
    // is a cursed win and has a cursed winning capture, or the position
    // has an ep capture as only best move.
    // This is used in probe_dtz().
    //
    // The return value is from the point of view of the side to move:
    // -2 : loss
    // -1 : loss, but draw under 50-move rule
    //  0 : draw
    //  1 : win, but draw under 50-move rule
    //  2 : win
    internal unsafe int probe_wdl(in FathomPos pos, out int success)
    {
      success = 1;

      // Generate (at least) all legal captures including (under)promotions.
      TBMoveList moves = gen_captures(in pos);
      int bestCap = -3, bestEp = -3;

      // We do capture resolution, letting bestCap keep track of the best
      // capture without ep rights and letting bestEp keep track of still
      // better ep captures if they exist.
      for (int i = 0; i < moves.NumMoves; i++)
      {
        FathomPos pos1 = default;
        ushort move = moves.Moves[i];
        if (!is_capture(pos, move))
          continue;
        if (!do_move(ref pos1, in pos, move))
        {
          continue; // illegal move
        }

        int vx = -probe_ab(in pos1, -2, -bestCap, ref success);
        if (success == 0) return 0;
        if (vx > bestCap)
        {
          if (vx == 2)
          {
            success = 2;
            return 2;
          }
          if (!is_en_passant(pos, move))
            bestCap = vx;
          else if (vx > bestEp)
            bestEp = vx;
        }
      }


      int v = probe_wdl_table(pos, ref success);
      if (success == 0)
      {
        return 0;
      }

      // Now max(v, bestCap) is the WDL value of the position without ep rights.
      // If the position without ep rights is not stalemate or no ep captures
      // exist, then the value of the position is max(v, bestCap, bestEp).
      // If the position without ep rights is stalemate and bestEp > -3,
      // then the value of the position is bestEp (and we will have v == 0).

      if (bestEp > bestCap)
      {
        if (bestEp > v)
        { // ep capture (possibly cursed losing) is best.
          success = 2;
          return bestEp;
        }
        bestCap = bestEp;
      }

      // Now max(v, bestCap) is the WDL value of the position unless
      // the position without ep rights is stalemate and bestEp > -3.

      if (bestCap >= v)
      {
        // No need to test for the stalemate case here: either there are
        // non-ep captures, or bestCap == bestEp >= v anyway.
        success = 1 + ((bestCap > 0) ? 1 : 0);
        return bestCap;
      }

      // Now handle the stalemate case.
      if (bestEp > -3 && v == 0)
      {
        TBMoveList moves1 = gen_moves(in pos);

        // Check for stalemate in the position with ep captures.
        bool foundMove = false;
        for (int i = 0; i < moves.NumMoves; i++)
        {
          ushort move = moves.Moves[i];

          if (!is_en_passant(pos, move) && legal_move(pos, move))
          {
            foundMove = true;
            break;
          }
        }

        if (!foundMove && !is_check(pos))
        {
          // stalemate score from tb (w/o e.p.), but an en-passant capture
          // is possible.
          success = 2;
          return bestEp;
        }
      }
      // Stalemate / en passant not an issue, so v is the correct value.

      return v;
    }


    static readonly int[] wdl_to_dtz = new[] { -1, -101, 0, 101, 1 };

    // Probe the DTZ table for a particular position.
    // If *success != 0, the probe was successful.
    // The return value is from the point of view of the side to move:
    //         n < -100 : loss, but draw under 50-move rule
    // -100 <= n < -1   : loss in n ply (assuming 50-move counter == 0)
    //         0        : draw
    //     1 < n <= 100 : win in n ply (assuming 50-move counter == 0)
    //   100 < n        : win, but draw under 50-move rule
    //
    // If the position mate, -1 is returned instead of 0.
    //
    // The return value n can be off by 1: a return value -n can mean a loss
    // in n+1 ply and a return value +n can mean a win in n+1 ply. This
    // cannot happen for tables with positions exactly on the "edge" of
    // the 50-move rule.
    //
    // This means that if dtz > 0 is returned, the position is certainly
    // a win if dtz + 50-move-counter <= 99. Care must be taken that the engine
    // picks moves that preserve dtz + 50-move-counter <= 99.
    //
    // If n = 100 immediately after a capture or pawn move, then the position
    // is also certainly a win, and during the whole phase until the next
    // capture or pawn move, the inequality to be preserved is
    // dtz + 50-movecounter <= 100.
    //
    // In short, if a move is available resulting in dtz + 50-move-counter <= 99,
    // then do not accept moves leading to dtz + 50-move-counter == 100.
    //
    internal int probe_dtz(in FathomPos pos, out int success)
    {
      int wdl = probe_wdl(pos, out success);
      if (success == 0)
      {
        return 0;
      }

      // If draw, then dtz = 0.
      if (wdl == 0) return 0;

      // Check for winning capture or en passant capture as only best move.
      if (success == 2)
      {
        return wdl_to_dtz[wdl + 2];
      }

      TBMoveList moves = null;
      FathomPos pos1 = default;

      // If winning, check for a winning pawn move.
      if (wdl > 0)
      {
        // Generate at least all legal non-capturing pawn moves
        // including non-capturing promotions.
        // (The following call in fact generates all moves.)


        // 6 seconds for 20mm move generations?
        {
          moves = gen_legal(in pos);
        }

        for (int i = 0; i < moves.NumMoves; i++)
        {
          ushort move = moves.Moves[i];
          if (type_of_piece_moved(in pos, move) != (int)FathomPieceType.PAWN || is_capture(pos, move))
          {
            continue;
          }

          if (!do_move(ref pos1, in pos, move))
          {
            continue; // not legal
          }

          int v = -probe_wdl(in pos1, out success);
          if (success == 0)
          {
            return 0;
          }

          if (v == wdl)
          {
            Debug.Assert(wdl < 3);
            return wdl_to_dtz[wdl + 2];
          }
        }
      }

      // If we are here, we know that the best move is not an ep capture.
      // In other words, the value of wdl corresponds to the WDL value of
      // the position without ep rights. It is therefore safe to probe the
      // DTZ table with the current value of wdl.

      int dtz = probe_dtz_table(pos, wdl, ref success);
      if (success >= 0)
        return wdl_to_dtz[wdl + 2] + ((wdl > 0) ? dtz : -dtz);

      // *success < 0 means we need to probe DTZ for the other side to move.
      int best;
      if (wdl > 0)
      {
        best = int.MaxValue; ;
      }
      else
      {
        // If (cursed) loss, the worst case is a losing capture or pawn move
        // as the "best" move, leading to dtz of -1 or -101.
        // In case of mate, this will cause -1 to be returned.
        best = wdl_to_dtz[wdl + 2];

        // If wdl < 0, we still have to generate all moves.
        moves = gen_moves(in pos);
      }

      for (int i = 0; i < moves.NumMoves; i++)
      {
        ushort move = moves.Moves[i];

        // We can skip pawn moves and captures.
        // If wdl > 0, we already caught them. If wdl < 0, the initial value
        // of best already takes account of them.
        if (is_capture(pos, move) || type_of_piece_moved(pos, move) == (int)FathomPieceType.PAWN)
          continue;

        if (!do_move(ref pos1, in pos, move))
        {
          // move was not legal
          continue;
        }
        int v = -probe_dtz(in pos1, out success);
        // Check for the case of mate in 1
        if (v == 1 && is_mate(in pos1))
          best = 1;
        else if (wdl > 0)
        {
          if (v > 0 && v + 1 < best)
            best = v + 1;
        }
        else
        {
          if (v - 1 < best)
            best = v - 1;
        }
        if (success == 0)
        {
          return 0;
        }
      }

      return best;
    }

    static readonly int[] WdlToRank = new int[] { -1000, -899, 0, 899, 1000 };
    static readonly int[] WdlToValue = new int[] {
    -TB_VALUE_MATE + TB_MAX_MATE_PLY + 1,
    TB_VALUE_DRAW - 2,
    TB_VALUE_DRAW,
    TB_VALUE_DRAW + 2,
    TB_VALUE_MATE - TB_MAX_MATE_PLY - 1
  };

    // Use the WDL tables to rank all root moves in the list.
    // This is a fallback for the case that some or all DTZ tables are missing.
    // A return value of 0 means that not all probes were successful.
    int root_probe_wdl(in FathomPos pos, bool useRule50, List<TbRootMove> rm)
    {
      int v, success;

      // Probe, rank and score each move.
      TBMoveList moves = gen_legal(in pos);
      FathomPos pos1 = default;
      for (int i = 0; i < moves.NumMoves; i++)
      {
        TbRootMove m = new TbRootMove();
        m.move = moves.Moves[i];
        do_move(ref pos1, in pos, m.move);
        v = -probe_wdl(in pos1, out success);

        if (success == 0)
        {
          return 0;
        }

        if (!useRule50)
        {
          v = v > 0 ? 2 : v < 0 ? -2 : 0;
        }
        m.tbRank = WdlToRank[v + 2];
        m.tbScore = WdlToValue[v + 2];
        rm.Add(m);
      }

      return 1;
    }

    // Use the DTZ tables to rank and score all root moves in the list.
    // A return value of 0 means that not all probes were successful.
    int root_probe_dtz(in FathomPos pos, bool hasRepeated, bool useRule50, List<TbRootMove> rm)
    {
      int v, success;

      // Obtain 50-move counter for the root position.
      int cnt50 = pos.rule50;

      // The border between draw and win lies at rank 1 or rank 900, depending
      // on whether the 50-move rule is used.
      int bound = useRule50 ? 900 : 1;

      // Probe, rank and score each move.
      TBMoveList rootMoves = gen_legal(in pos);
      FathomPos pos1 = default;
      for (int i = 0; i < rootMoves.NumMoves; i++)
      {
        TbRootMove m = new();
        m.move = rootMoves.Moves[i];
        do_move(ref pos1, in pos, m.move);

        // Calculate dtz for the current move counting from the root position.
        if (pos1.rule50 == 0)
        {
          // If the move resets the 50-move counter, dtz is -101/-1/0/1/101.
          v = -probe_wdl(in pos1, out success);
          Debug.Assert(v < 3);
          v = wdl_to_dtz[v + 2];
        }
        else
        {
          // Otherwise, take dtz for the new position and correct by 1 ply.
          v = -probe_dtz(in pos1, out success);
          if (v > 0) v++;
          else if (v < 0) v--;
        }
        // Make sure that a mating move gets value 1.
        if (v == 2 && is_mate(in pos1))
        {
          v = 1;
        }

        if (success == 0)
        {
          return 0;
        }

        // Better moves are ranked higher. Guaranteed wins are ranked equally.
        // Losing moves are ranked equally unless a 50-move draw is in sight.
        // Note that moves ranked 900 have dtz + cnt50 == 100, which in rare
        // cases may be insufficient to win as dtz may be one off (see the
        // comments before TB_probe_dtz()).
        int r = v > 0 ? (v + cnt50 <= 99 && !hasRepeated ? 1000 : 1000 - (v + cnt50))
               : v < 0 ? (-v * 2 + cnt50 < 100 ? -1000 : -1000 + (-v + cnt50))
               : 0;
        m.tbRank = r;

        // Determine the score to be displayed for this move. Assign at least
        // 1 cp to cursed wins and let it grow to 49 cp as the position gets
        // closer to a real win.
        m.tbScore = r >= bound ? TB_VALUE_MATE - TB_MAX_MATE_PLY - 1
                    : r > 0 ? Math.Max(3, r - 800) * TB_VALUE_PAWN / 200
                    : r == 0 ? TB_VALUE_DRAW
                    : r > -bound ? Math.Max(-3, r + 800) * TB_VALUE_PAWN / 200
                    : -TB_VALUE_MATE + TB_MAX_MATE_PLY + 1;
        rm.Add(m);
      }
      return 1;
    }



    internal static bool TEMPis_mate(in FathomPos pos)
    {
      if (!is_check(pos))
      {
        return false;
      }

      TBMoveList moves = gen_moves(pos);
      for (int i = 0; i < moves.NumMoves; i++)
      {
        FathomPos pos1 = default;
        if (do_move(ref pos1, in pos, moves.Moves[i]))
        {
          return false;
        }
      }

      return true;
    }

    // This supports the original Fathom root probe API
    ushort probe_root(in FathomPos pos, out int score, uint[] results)
    {
      score = default;

      int success = 0;
      int dtz = probe_dtz(pos, out success);
      if (success == 0)
      {
        return 0;
      }


      Span<short> scores = stackalloc short[MAX_MOVES];
      TBMoveList moves0 = gen_moves(in pos);

      int num_draw = 0;
      int j = 0;
      for (int i = 0; i < moves0.NumMoves; i++)
      {
        FathomPos pos1 = default;
        if (!do_move(ref pos1, in pos, moves0.Moves[i]))
        {
          scores[i] = SCORE_ILLEGAL;
          continue;
        }
        int v = 0;
        if (dtz > 0 && is_mate(in pos1))
        {
          v = 1;
        }
        else
        {
          if (pos1.rule50 != 0)
          {
            v = -probe_dtz(in pos1, out success);
            if (v > 0)
            {
              v++;
            }
            else if (v < 0)
            {
              v--;
            }
          }
          else
          {
            v = -probe_wdl(in pos1, out success);
            v = wdl_to_dtz[v + 2];
          }
        }
        num_draw += (v == 0) ? 1 : 0;
        if (success == 0)
        {
          return 0;
        }

        scores[i] = (short)v;
        if (results != null)
        {
          uint res = 0;
          res = TB_SET_WDL(res, (int)dtz_to_wdl(pos.rule50, v));
          res = TB_SET_FROM(res, move_from(moves0.Moves[i]));
          res = TB_SET_TO(res, move_to(moves0.Moves[i]));
          res = TB_SET_PROMOTES(res, move_promotes(moves0.Moves[i]));
          res = TB_SET_EP(res, is_en_passant(pos, moves0.Moves[i]) ? (uint)1 : 0);
          res = TB_SET_DTZ(res, (uint)(v < 0 ? -v : v));
          results[j++] = (uint)res;
        }
      }

      if (results != null)
      {
        results[j++] = TB_RESULT_FAILED;
      }

      score = dtz;

      // Now be a bit smart about filtering out moves.
      if (dtz > 0)        // winning (or 50-move rule draw)
      {
        int best = BEST_NONE;
        ushort best_move = 0;
        for (int i = 0; i < moves0.NumMoves; i++)
        {
          int v = scores[i];
          if (v == SCORE_ILLEGAL)
            continue;
          if (v > 0 && v < best)
          {
            best = v;
            best_move = moves0.Moves[i];
          }
        }
        return (ushort)(best == BEST_NONE ? 0 : best_move);
      }
      else if (dtz < 0)   // losing (or 50-move rule draw)
      {
        int best = 0;
        ushort best_move = 0;
        for (int i = 0; i < moves0.NumMoves; i++)
        {
          int v = scores[i];
          if (v == SCORE_ILLEGAL)
          {
            continue;
          }

          if (v < best)
          {
            best = v;
            best_move = (ushort)moves0.Moves[i];
          }
        }
        return (ushort)(best == 0 ? MOVE_CHECKMATE : best_move);
      }
      else // drawing
      {
        // Check for stalemate:
        if (num_draw == 0)
        {
          return MOVE_STALEMATE;
        }

        // Select a "random" move that preserves the draw.
        // Uses calc_key as the PRNG.
        //int count = (int)calc_key(pos, !pos.turn) % num_draw;
        ulong count = (ulong)(calc_key(pos, !pos.turn) % (ulong)num_draw);

        for (int i = 0; i < moves0.NumMoves; i++)
        {
          int v = scores[i];
          if (v == SCORE_ILLEGAL)
          {
            continue;
          }
          if (v == 0)
          {
            if (count == 0)
            {
              return (ushort)moves0.Moves[i];
            }
            count--;
          }
        }
        return 0;
      }

    }

    // Count number of placements of k like pieces on n squares
    static ulong subfactor(ulong k, ulong n)
    {
      ulong f = n;
      ulong l = 1;
      for (ulong i = 1; i < k; i++)
      {
        f *= n - i;
        l *= i + 1;
      }

      return f / l;
    }

    #endregion


    #region Little/Big Endian

    static unsafe uint read_le_u32(void* p)
    {
      return from_le_u32(*(uint*)p);
    }

    //  static unsafe ushort read_le_u16(void* p)
    //  {
    //    return from_le_u16(*(ushort*)p);
    //  }

    static ushort from_le_u16(ushort x)
    {
      return x;
    }

    static ulong from_be_u64(ulong x)
    {
      return ((x & 0xff00000000000000UL) >> 56) |
        ((x & 0x00ff000000000000UL) >> 40) |
        ((x & 0x0000ff0000000000UL) >> 24) |
        ((x & 0x000000ff00000000UL) >> 8) |
        ((x & 0x00000000ff000000UL) << 8) |
        ((x & 0x0000000000ff0000UL) << 24) |
        ((x & 0x000000000000ff00UL) << 40) |
        ((x & 0x00000000000000ffUL) << 56);
    }


    static uint from_be_u32(uint x)
    {
      return ((x & 0xff000000) >> 24) |
             ((x & 0x00ff0000) >> 8) |
            ((x & 0x0000ff00) << 8) |
            ((x & 0x000000ff) << 24);
    }

    static uint from_le_u32(uint x)
    {
      return x;
    }



    #endregion

  }

}