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

#endregion


namespace Ceres.Chess
{
  /// <summary>
  /// Result of game expected from this node if game were played out with best play by both sides.
  /// 
  /// Warning: do not mofify the order of these values since 
  /// this particular order is assumed by the IsTerminalMethod.
  /// </summary>
  public enum GameResult : byte 
  { 
    /// <summary>
    /// State before being initialized
    /// </summary>
    NotInitialized,

    /// <summary>
    /// Not (yet) known what result of game would be under best play
    /// </summary>
    Unknown, 

    /// <summary>
    /// Draw by stalemate
    /// </summary>
    Draw, 

    /// <summary>
    /// Side to move is checkmated
    /// </summary>
    Checkmate 
  };
}


