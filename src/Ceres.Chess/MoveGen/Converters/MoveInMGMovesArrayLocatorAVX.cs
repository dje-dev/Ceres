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
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endregion


namespace Ceres.Chess.MoveGen.Converters
{
  /// <summary>
  /// AVX version of FindMoveIndex.
  /// See FindMoveIndex.linq test code which seems to show this AVX version is considerably faster.
  /// (however it was actually found to run slower in Ceres for unknown reasons).
  /// </summary>
  public static class MoveInMGMovesArrayLocatorAVX
  {
    const int ThreePow0 = 3;
    const int ThreePow1 = ThreePow0 * 16;
    const int ThreePow2 = ThreePow1 * 16;
    const int ThreePow3 = ThreePow2 * 16;
    const int ThreePow4 = ThreePow3 * 16;
    const int ThreePow5 = ThreePow4 * 16;
    const int ThreePow6 = ThreePow5 * 16;
    const int ThreePow7 = ThreePow6 * 16;


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static unsafe int FindIndexOfShortAtEvenIndexHavingValue(Span<int> data, short searchValue, int startIndex, int maxIndex)
    {
      // For convenience/efficiency we require arrays to be divisible by 8
      Debug.Assert(data.Length % 8 == 0);

      Span<short> dTargetValue = stackalloc short[] {
        searchValue,
        short.MinValue,
        searchValue,
        short.MinValue,
        searchValue,
        short.MinValue,
        searchValue,
        short.MinValue,
        searchValue,
        short.MinValue,
        searchValue,
        short.MinValue,
        searchValue,
        short.MinValue,
        searchValue,
        short.MinValue
      };


      int numBlocksProcessed = startIndex / 8;
      int maskMove = 0;

      fixed (short* pTargetValue = &dTargetValue[0])
      fixed (int* pStartData = &data[0])
      {
        short* pStartDataShort = (short*)pStartData;
        Vector256<short> targetData = Avx.LoadVector256(pTargetValue);

        for (int i = numBlocksProcessed * 8 * 2; i < data.Length * 2; i += 8 * 2)
        {
          // Load this set of values to examine
          Vector256<short> vValues = Avx.LoadVector256(&pStartDataShort[i]);

          // Compare for equality
          Vector256<short> vEQ = Avx2.CompareEqual(vValues, targetData);

          // Get resulting equality mask so we can tell which index within block had target value
          Vector256<byte> equalityAsBytes = vEQ.As<short, byte>();
          maskMove = Avx2.MoveMask(equalityAsBytes);
          if (maskMove != 0) break;

          numBlocksProcessed++;
        }


        // Translate  mask into which index
        int indexInBlock;
        if (maskMove <= ThreePow3)
        {
          indexInBlock = maskMove switch
          {
            ThreePow0 => 0,
            ThreePow1 => 1,
            ThreePow2 => 2,
            ThreePow3 => 3,
            _ => -2 // false accidental match at odd index
          };
        }
        else
        {
          indexInBlock = maskMove switch
          {
            ThreePow4 => 4,
            ThreePow5 => 5,
            ThreePow6 => 6,
            ThreePow7 => 7,
            _ => -2 // false accidental match at odd index
          };
        }

        if (indexInBlock == -2) return -2;

        int index = numBlocksProcessed * 8 + indexInBlock;

        if (index < startIndex || index > maxIndex)
          return -1;
        else
          return index;
      }


      Console.WriteLine(maskMove);

    }

    // --------------------------------------------------------------------------------------------
    internal unsafe static int FindMoveIndexAVX(MGMove[] moves, ConverterMGMoveEncodedMove.FromTo moveSquares, int startIndex, int numMovesUsed)
    {
      // Use non-AVX version if few move to check
      if (numMovesUsed - startIndex < 12) return MoveInMGMovesArrayLocator.FindMoveIndex(moves, moveSquares, startIndex, numMovesUsed);

      void* ptrStart = Unsafe.AsPointer(ref moves[0]);
      Span<int> ints = new Span<int>(ptrStart, numMovesUsed);

      int foundIndex = FindIndexOfShortAtEvenIndexHavingValue(ints, moveSquares.FromAndToCombined, startIndex, numMovesUsed - 1);

      if (foundIndex == -2)
      {
        // The AVX routine found a false pseudo-match at odd index
        // Fallback to non AVX version 
        // This is expected to happen only very rarely
        return MoveInMGMovesArrayLocator.FindMoveIndex(moves, moveSquares, startIndex, numMovesUsed);
      }

      return foundIndex;
    }
  }

}
