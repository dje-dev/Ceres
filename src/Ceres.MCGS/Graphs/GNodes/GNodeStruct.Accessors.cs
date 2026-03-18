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
using System.Reflection;
using System.Runtime.InteropServices;
using System.Diagnostics;

using Ceres.Chess;
using Ceres.Base.DataTypes;
using Ceres.Chess.MoveGen;
using System.Runtime.CompilerServices;
using Ceres.MCGS.Graphs.GNodes;

#endregion

namespace Ceres.MCGS.Graphs.GNodes;

public unsafe partial struct GNodeStruct
{
  /// <summary>
  /// Game terminal status
  /// </summary>
  public GameResult Terminal
  {
    readonly get => miscFields.Terminal;
    set => miscFields.Terminal = value;
  }

  /// <summary>
  ///  Moves left estimate for this position
  /// </summary>
  public byte M
  {
    readonly get => MRaw;
    set => MRaw = Math.Min(MAX_M, value);
  }


  /// <summary>
  /// If the node is the root node of the graph.
  /// </summary>
  public bool IsGraphRoot
  {
    readonly get => miscFields.IsGraphRoot;
    set => miscFields.IsGraphRoot = value;
  }

  /// <summary>
  /// If the node is the root node of the currently active search.
  /// </summary>
  public bool IsSearchRoot
  {
    readonly get => miscFields.IsSearchRoot;
    set => miscFields.IsSearchRoot = value;
  }

  /// <summary>
  /// The Move50CategoryEnum assoicated with the position.
  /// </summary>
  public Move50CategoryEnum Move50CategoryEnum
  {
    readonly get => miscFields.Move50Category;
    set => miscFields.Move50Category = value;
  }

  /// <summary>
  /// Unused boolean field.
  /// </summary>
  public bool UnusedBool1
  {
    readonly get => miscFields.UnusedBool1;
    set => miscFields.UnusedBool1 = value;
  }



#if NOT
/// <summary>
  /// The 64 bit Zobrist hash is used to find nodes in the transposition table
  /// within the same hash equivlance class. However hash collisions will
  /// occur (perhaps ever 300 million to 3 billion positions) and establishing
  /// incorrect linkages could lead to incorrect valuations or invalid move lists
  /// being propagated to the linked nodes.
  /// 
  /// The HashCrossheck is an independent 8 bit hash value used 
  /// as an additional crosscheck for equality before establishing linkages
  /// to transposition nodes to greatly reduce their likelihood.
  /// </summary>
  public byte HashCrosscheck;
#endif

  /// <summary>
  /// If at least one of the children has been found to 
  /// be a checkmate (terminal node).
  /// </summary>
  public bool CheckmateKnownToExistAmongChildren
  {
    readonly get => miscFields.CheckmateKnownToExistAmongChildren;
    set => miscFields.CheckmateKnownToExistAmongChildren = value;
  }

  /// <summary>
  /// If at least one of the children has been found to 
  /// be a draw (terminal node).
  /// </summary>
  public bool DrawKnownToExistAmongChildren
  {
    readonly get => miscFields.DrawKnownToExistAmongChildren;
    set => miscFields.DrawKnownToExistAmongChildren = value;
  }

  /// <summary>
  /// Number of pieces on the board.
  /// </summary>
  public byte NumPieces
  {
    readonly get => miscFields.NumPieces;
    set => miscFields.NumPieces = value;
  }

  /// <summary>
  /// Number of pawns still on their starting square.
  /// </summary>
  public byte NumRank2Pawns
  {
    readonly get => miscFields.NumRank2Pawns;
    set => miscFields.NumRank2Pawns = value;
  }


  /// <summary>
  /// Nodes active in current tree have generation 0.
  /// Nodes with higher generations can exist when tree reuse is enabled,
  /// and indicate nodes left behind by prior search (but no longer valid).
  /// The generation number indicates how many moves prior the node was last active.
  /// TODO: currently we only maintain these values if enabled TREE_REUSE_RETAINED_POSITION_CACHE_ENABLED
  /// </summary>
  public bool IsOldGeneration
  {
    readonly get => miscFields.IsOldGeneration;
    set => miscFields.IsOldGeneration = value;
  }

  public Move50CategoryEnum Move50Category
  {
    readonly get => miscFields.Move50Category;
    set => miscFields.Move50Category = value;
  }

  public bool HasRepetitions
  {
    readonly get => miscFields.HasRepetitions;
    set => miscFields.HasRepetitions = value;
  }

  public NodeIndex ParentIndex
  {
    readonly get => throw new NotImplementedException();
    set => throw new NotImplementedException();
  }



  // TODO: move into ObjUtils
  public static void DumpFieldsAndProperties<T>(T data, int indentChars = 0) where T : struct
  {
    string indentString = new(' ', indentChars);
    Type type = typeof(T);
    Console.WriteLine($"{indentString}Type: {type.Name}, Size: {Unsafe.SizeOf<T>()} bytes");

    FieldInfo[] fields = type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
    foreach (FieldInfo field in fields)
    {
      if (data.Equals(new T()))
      {
        Console.WriteLine($"{indentString}{field.FieldType.Name,-10}  {field.GetType(),20}   {field.Name,-45}");
      }
      else
      { 
        Console.WriteLine($"{indentString}{field.FieldType.Name,-10}  {field.GetType(),20}  {field.GetValue(data),20}   {field.Name,-45}");
      }
    }

    PropertyInfo[] properties = type.GetProperties(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
    foreach (PropertyInfo property in properties)
    {
      object value = property.GetGetMethod(true)?.Invoke(data, null);
      Console.WriteLine($"{indentString}{property.PropertyType.Name,-10}  {value,20}   {property.Name,-45}");
    }
  }


  // TODO: move into ObjUtils


  public static void DumpStructFields<T>(T data) where T : unmanaged
  {
    Console.WriteLine();
    Console.WriteLine(typeof(T).Name + " (" + sizeof(T) + " bytes)");
    // Get both public and non-public instance fields
    FieldInfo[] fields = typeof(T).GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

    // Determine the maximum field name length for alignment
    int maxFieldNameLength = 0;
    foreach (FieldInfo field in fields)
    {
      if (field.Name.Length > maxFieldNameLength)
      {
        maxFieldNameLength = field.Name.Length;
      }
    }

    // Initialize running total of field sizes
    int runningTotalSize = 0;

    // Print the fields and their values
    Console.WriteLine($"  {"Field".PadRight(maxFieldNameLength)} | {"Value",-20} | {"Type",10} | {"Size (bytes)",15} | {"Running Total",15}");
    Console.WriteLine(new string('-', maxFieldNameLength + 65)); // Adjust the total width as necessary
    foreach (FieldInfo field in fields)
    {
      if (!field.IsStatic)
      {
        string name = field.Name;
        string value = field.GetValue(data).ToString();
        value = value[..Math.Min(18, value.ToString().Length)];
        string type = field.FieldType.Name[..Math.Min(10, field.FieldType.Name.Length)];
        int size = Marshal.SizeOf(field.FieldType);
        runningTotalSize += size;
        Console.WriteLine($"  {name.PadRight(maxFieldNameLength)} | {value,-20} | {type,10} | {size,15} bytes | {runningTotalSize,15} bytes");
      }
    }
  }

  public readonly void DumpRawFields()
  {
    DumpStructFields(this);
    Console.WriteLine("MISC FIELDS");
    DumpFieldsAndProperties(miscFields, 2);
  }


  public readonly void Dump()
  {
    Console.WriteLine($"ReuseGenerationNum {IsOldGeneration}");
    Console.WriteLine($"ZobristHash {HashStandalone}");

    //      Console.WriteLine($"P {P,10:F3}  MPosition          {MPosition,10:F3}");
    Console.WriteLine($"NumPolicyMoves                  {NumPolicyMoves,-4}  NumChildrenExpanded {NumEdgesExpanded,-4} NumChildrenExpanded {NumEdgesExpanded,-4}");
    Console.WriteLine($"ChildStartBlockIndex ");//           {ChildInfo}");


    Console.WriteLine($"WinP {WinP,10:F3} LossP {LossP,10:F3}");
    Console.WriteLine($"Q    {Q,10:F4}");
    Console.WriteLine($"DSum/N (DAvg)    {D / N,10:F4}");
    Console.WriteLine($"Terminal         {Terminal}");
    Console.WriteLine($"DrawCanBeClaimed {DrawKnownToExistAmongChildren}");

  }
}