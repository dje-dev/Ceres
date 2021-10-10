#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

//#define SMALL

#region License

/* 
License Note
   
This code originated from Github repository from Judd Niemann
and is licensed with the MIT License.

This version is modified by David Elliott, including a translation to C# and 
some moderate modifications to improve performance and modularity.
*/

/*

MIT License

Copyright(c) 2016-2017 Judd Niemann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#region Using directives


#endregion


#endregion

#region Using directives

using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

#endregion

namespace Ceres.Chess.MoveGen
{
  /// <summary>
  /// 
  /// TODO: Perhaps make this a struct?
  /// </summary>
  [Serializable]
  public class MGMoveList : IEnumerable<MGMove>
  {
    /// <summary>
    /// Initial allocation of 54 moves is usually sufficient
    /// </summary>
    const int INITIAL_ALLOC_MOVES = 54;

    /// <summary>
    /// Array of moves.
    /// </summary>
    public MGMove[] MovesArray;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="maxMoves"></param>
    public MGMoveList(int maxMoves = INITIAL_ALLOC_MOVES) 
    {
      MovesArray = GC.AllocateUninitializedArray<MGMove>(maxMoves);
      NumMovesUsed = 0; 
    }

    public MGMoveList(MGMove[] moveArrayBuffer)
    {
      MovesArray = moveArrayBuffer;
      NumMovesUsed = 0;
    }

    /// <summary>
    /// Copy constructor.
    /// </summary>
    /// <param name="other"></param>
    public MGMoveList(MGMoveList other)
    {
      NumMovesUsed = other.NumMovesUsed;

      // Allocate and initialize move array of exactly correct size
      MovesArray = GC.AllocateUninitializedArray<MGMove>(other.NumMovesUsed);
      Array.Copy(other.MovesArray, MovesArray, NumMovesUsed);
    }


    /// <summary>
    /// Copies moves from another MGMoveList.
    /// </summary>
    /// <param name="other"></param>
    public void Copy(MGMoveList other)
    {
      if (MovesArray.Length < other.NumMovesUsed)
      {
        // Need an enlarged array
        MovesArray = GC.AllocateUninitializedArray<MGMove>(other.NumMovesUsed);
      }
      NumMovesUsed = other.NumMovesUsed;
      Array.Copy(other.MovesArray, MovesArray, other.NumMovesUsed);
    }


    /// <summary>
    /// Expands MoveArray if necessary to accomodate numMoves additional moves
    /// </summary>
    /// <param name="numMoves"></param>
    /// 
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void InsureMoveArrayHasRoom(int numMoves)
    {
      // Note that we have to leave room for extra sentinel value at end (NumMovesUsed + 1)
      int needArraySize = (NumMovesUsed + 1 + numMoves);
      if (needArraySize > MovesArray.Length)
        DoEnlargeArray(needArraySize);
    }

    internal void DoEnlargeArray(int needArraySize)
    {
      const int EXTRA_ALLOC_MOVES = 8; // allocate more than bare minimum to reduce number of times we have to expand
      MGMove[] newMoveArray = GC.AllocateUninitializedArray<MGMove>(needArraySize + EXTRA_ALLOC_MOVES);
      Array.Copy(MovesArray, newMoveArray, NumMovesUsed);
      MovesArray = newMoveArray;
    }

    /// <summary>
    /// Clears to empty state.
    /// </summary>
    public void Clear() => NumMovesUsed = 0;
    
    /// <summary>
    /// Resizes underlying move array to same size as actual number of moves
    /// </summary>
    public void Resize() => Array.Resize<MGMove>(ref MovesArray, NumMovesUsed);
    

#if SMALL
//    Span<int> first = stackalloc int[3] { 1, 2, 3 };
    public Span<ulong> MovesArray = stackalloc ulong[128];
    //private fixed ulong MovesArray[128];
#else
#endif
    //    public ChessMove this[int index] { get { return Moves[index]; } set { Moves[index] = value; } }

    public int NumMovesUsed;

    static MGMove FromULong(ulong value) => default;
    static ulong ToLong(MGMove move) => default;

    public bool IsEmpty => NumMovesUsed == 0;

    /// <summary>
    /// Returns the index of a specified move (starting search at specified index), or -1 if not found.
    /// Optimized for performance.
    /// </summary>
    /// <param name="moves"></param>
    /// <param name="move"></param>
    /// <param name="startIndex"></param>
    /// <returns></returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int FindIndex(MGMove move, int startIndex)
    {
      int numMovesUsed = NumMovesUsed;
      byte moveToSquare = move.ToSquareIndex;

      for (int i = startIndex; i < numMovesUsed; i++)
        if (MovesArray[i].ToSquareIndex == moveToSquare)
          if (MovesArray[i].Equals(move))
            return i;
      return -1;
    }


    public IEnumerator<MGMove> GetEnumerator()
    {
#if SMALL
      for (int i = 0; i < NumMovesUsed; i++)
        yield return FromULong(MovesArray[i]);

#else
      for (int i = 0; i < NumMovesUsed; i++)
        yield return MovesArray[i];
#endif
    }


    IEnumerator IEnumerable.GetEnumerator()
    {
      throw new NotImplementedException();
      for (int i = 0; i < NumMovesUsed; i++)
        yield return MovesArray[i];
    }

    public override string ToString()
    {
      string sideStr = NumMovesUsed == 0 ? "" : (MovesArray[0].BlackToMove ? "Black" : "White");
      return $"<MGMoveList ({NumMovesUsed} { sideStr }";
    }

  }
}

