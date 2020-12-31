#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

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
#endregion

#region Using Directives

using System;
using System.Runtime.CompilerServices;

using HashKey = System.UInt64;
using ZobristKey = System.UInt64;

#endregion

namespace Ceres.Chess.MoveGen
{
  public static class MGZobristKeySet
  {
    public static ZobristKey[][] zkPieceOnSquare;
	  public static ZobristKey zkBlackToMove;
    public static ZobristKey zkWhiteCanCastle;
    public static ZobristKey zkWhiteCanCastleLong;
    public static ZobristKey zkBlackCanCastle;
    public static ZobristKey zkBlackCanCastleLong;

    // pre-fabricated combinations of keys for castling:
    public static ZobristKey zkDoBlackCastle;
    public static ZobristKey zkDoBlackCastleLong;
    public static ZobristKey zkDoWhiteCastle;
    public static ZobristKey zkDoWhiteCastleLong;

		[ModuleInitializer]
		internal static void ClassInitialize()
		{
      // initialize arrays
      zkPieceOnSquare = new ZobristKey[16][];
      for (int i = 0; i < 16; i++) zkPieceOnSquare[i] = new ZobristKey[64];

      // 
      // Create a Random Number Generator, using the 64-bit Mersenne Twister Algorithm
      // with a uniform distribution of ints 
      //      std::random_device rd;
      //      std::mt19937_64 e2(rd());
      //      std::uniform_int_distribution < unsigned long long int> dist(0, 0xffffffffffffffff);

      Random rand = new Random(713385737);
      //      Random rand = new Random(1337);

      ulong dist() => (ulong)(rand.NextDouble() * ulong.MaxValue);
#if NO_BETTER
      RNGCryptoServiceProvider provider = new RNGCryptoServiceProvider();
      ulong dist()
      {
        var byteArray = new byte[8];
        provider.GetBytes(byteArray);
        ulong random = BitConverter.ToUInt64(byteArray, 0);
        return random;
      }
#endif

      for (int n = 0; n < zkPieceOnSquare.Length; ++n)
        for (int m = 0; m < 64; ++m)
          zkPieceOnSquare[n][m] = dist();

      zkBlackToMove = dist();
      zkWhiteCanCastle = dist();
      zkWhiteCanCastleLong = dist();
      zkBlackCanCastle = dist();
      zkBlackCanCastleLong = dist();

      // generate pre-fabricated castling keys:
      zkDoBlackCastle =
        zkPieceOnSquare[MGPositionConstants.BKING][59] ^ // Remove King from e8
        zkPieceOnSquare[MGPositionConstants.BKING][57] ^ // Place King on g8
        zkPieceOnSquare[MGPositionConstants.BROOK][56] ^ // Remove Rook from h8
        zkPieceOnSquare[MGPositionConstants.BROOK][58] ^ // Place Rook on f8
        zkBlackCanCastle;      // (unconditionally) flip black castling

      zkDoBlackCastleLong =
        zkPieceOnSquare[MGPositionConstants.BKING][59] ^ // Remove King from e8
        zkPieceOnSquare[MGPositionConstants.BKING][61] ^ // Place King on c8
        zkPieceOnSquare[MGPositionConstants.BROOK][63] ^ // Remove Rook from a8
        zkPieceOnSquare[MGPositionConstants.BROOK][60] ^ // Place Rook on d8
        zkBlackCanCastleLong;    // (unconditionally) flip black castling long

      zkDoWhiteCastle =
        zkPieceOnSquare[MGPositionConstants.WKING][3] ^ // Remove King from e1
        zkPieceOnSquare[MGPositionConstants.WKING][1] ^ // Place King on g1
        zkPieceOnSquare[MGPositionConstants.WROOK][0] ^ // Remove Rook from h1
        zkPieceOnSquare[MGPositionConstants.WROOK][2] ^ // Place Rook on f1
        zkWhiteCanCastle;      // (unconditionally) flip white castling

      zkDoWhiteCastleLong =
        zkPieceOnSquare[MGPositionConstants.WKING][3] ^ // Remove King from e1
        zkPieceOnSquare[MGPositionConstants.WKING][5] ^ // Place King on c1
        zkPieceOnSquare[MGPositionConstants.WROOK][7] ^ // Remove Rook from a1
        zkPieceOnSquare[MGPositionConstants.WROOK][4] ^ // Place Rook on d1
        zkWhiteCanCastleLong;    // (unconditionally) flip white castling long
    }
  }
}

