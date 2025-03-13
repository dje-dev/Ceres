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
using System.Collections.Generic;
using System.Linq;

#endregion

namespace Ceres.Base.DataTypes
{
  /// <summary>
  /// Efficiently maintains the set of "top N" entities as they are scanned,
  /// where order is defined by a provided delegate.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  public class TopN<T> : IEnumerable<T>
  {
    /// <summary>
    /// The number of entities to be tracked.
    /// </summary>
    public readonly int N;

    /// <summary>
    /// The mapping function used to evaluate the entities.
    /// </summary>
    public readonly Func<T, IComparable> Evaluator;

    /// <summary>
    /// Returns the number of entities currently being tracked.
    /// </summary>
    public int Count => numFound;

    #region Internal data 

    /// <summary>
    /// The current entity having the lowest value currently tracked
    /// </summary>
    T currentCutoffValue;

    /// <summary>
    /// Set of current top entities
    /// </summary>
    T[] tops;

    /// <summary>
    /// Number of entities currently being tracked (may be less than N at startup)
    /// </summary>
    int numFound;

    TopNComparer copyFunc;

    #endregion


    /// <summary>
    /// Constructor (including a set of initial entities to scan)
    /// </summary>
    /// <param name="items">set of initial T to be scanned</param>
    /// <param name="maxN">number of top entities to track</param>
    /// <param name="eval">evaluator function for ranking T</param>
    public TopN(IEnumerable<T> items, int maxN, Func<T, IComparable> eval)
    {
      N = maxN;
      Evaluator = eval ?? throw new ArgumentNullException(nameof(eval));

      tops = new T[maxN];
      copyFunc = new TopNComparer(eval);
      if (items != null)
      {
        Add(items);
      }
    }


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="maxN"></param>
    /// <param name="eval"></param>
    public TopN(int maxN, Func<T, IComparable> eval) : this(null, maxN, eval)
    {
    }



    /// <summary>
    /// Returns a List of the TopN members.
    /// </summary>
    public List<T> Members
    {
      get
      {
        List<T> ret = new List<T>(numFound);
        for (int i = 0; i < numFound; i++)
        {
          ret.Add(tops[i]);
        }
        return ret;
      }
    }


    /// <summary>
    /// Indexer
    /// </summary>
    /// <param name="index"></param>
    /// <returns></returns>
    public T this[int index] => tops[index];


    /// <summary>
    /// Scans entities for potential addition to TopN set.
    /// </summary>
    /// <param name="items">Items to scan for potential addition</param>
    public void Add(IEnumerable<T> items)
    {
      foreach (T item in items)
      {
        Add(item);
      }
    }


    /// <summary>
    /// Potentially adds another item to the TopN set.
    /// </summary>
    /// <param name="item"></param>
    /// <returns></returns>
    public bool Add(T item)
    {
      if (numFound < N)
      {
        // We are not full yet, definitely add this one
        tops[numFound++] = item;
        Array.Sort<T>(tops, 0, numFound, copyFunc);
        currentCutoffValue = tops[0];
        return true;
      }

      // Exit if we are too small (less than cutoff value)
      if (copyFunc.Compare(item, currentCutoffValue) <= 0) return false;

      // Add this as new item, resort
      tops[0] = item;
      Array.Sort<T>(tops, 0, numFound, copyFunc);
      currentCutoffValue = tops[0];

      return true;
    }


    /// <summary>
    /// Removes all items from the TopN set.
    /// </summary>
    public void Clear() => numFound = 0;
    

    /// <summary>
    /// Enumerates over the top N elements from Top1 to TopN.
    /// </summary>
    /// <returns></returns>
    public IEnumerator GetEnumerator()
    {
      for (int i = 0; i < numFound; i++)
        yield return tops[numFound - i - 1];
    }



    /// <summary>
    /// Enumerates over the top N elements from Top1 to TopN
    /// </summary>
    /// <returns></returns>
    IEnumerator<T> IEnumerable<T>.GetEnumerator()
    {
      for (int i = 0; i < numFound; i++)
      {
        yield return tops[numFound - i - 1];
      }
    }


    /// <summary>
    /// Returns a string representation of the top N entities.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      string str = $"<TopN {Count} : ";
      for (int i = 0; i < numFound && i < 5; i++)
      {
        str += Evaluator (tops[i]).ToString() + " ";
      }
      return str + ">";
    }


    #region Internals

    private class TopNComparer : IComparer<T>
    {
      readonly Func<T, IComparable> mapper;
      public TopNComparer(Func<T, IComparable> mapper)
      {
        this.mapper = mapper;
      }

      public int Compare(T t1, T t2) => mapper(t1).CompareTo(mapper(t2));
    }

    #endregion
  }

}
