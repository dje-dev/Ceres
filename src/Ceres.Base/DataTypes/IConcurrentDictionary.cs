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

#endregion

namespace Ceres.Base.DataTypes;

/// <summary>
/// Minimal concurrent dictionary interface for transposition table use.
/// Supports the subset of ConcurrentDictionary operations used by Graph and related classes.
/// </summary>
public interface IConcurrentDictionary<TKey, TValue> : IEnumerable<KeyValuePair<TKey, TValue>>
  where TKey : IEquatable<TKey>
{
  bool TryGetValue(TKey key, out TValue value);
  bool TryAdd(TKey key, TValue value);
  TValue this[TKey key] { set; }
  int Count { get; }
}
