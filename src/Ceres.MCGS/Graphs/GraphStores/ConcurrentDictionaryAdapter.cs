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
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using Ceres.Base.DataTypes;

#endregion

namespace Ceres.MCGS.Graphs.GraphStores;

/// <summary>
/// Thin adapter that wraps a ConcurrentDictionary to implement IConcurrentDictionary.
/// </summary>
public sealed class ConcurrentDictionaryAdapter<TKey, TValue>
  : IConcurrentDictionary<TKey, TValue>
  where TKey : IEquatable<TKey>
{
  readonly ConcurrentDictionary<TKey, TValue> inner;

  public ConcurrentDictionaryAdapter(int concurrencyLevel, int capacity)
  {
    inner = new ConcurrentDictionary<TKey, TValue>(concurrencyLevel, capacity);
  }

  public bool TryGetValue(TKey key, out TValue value) => inner.TryGetValue(key, out value);
  public bool TryAdd(TKey key, TValue value) => inner.TryAdd(key, value);
  public TValue this[TKey key] { set => inner[key] = value; }
  public int Count => inner.Count;

  public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator() => inner.GetEnumerator();
  IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
