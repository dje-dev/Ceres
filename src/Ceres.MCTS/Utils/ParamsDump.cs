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
using System.IO;
using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.MCTS.LeafExpansion;
using Ceres.MCTS.Managers.Limits;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.Utils
{
  /// <summary>
  /// Static utility class for dumping search parameters to Console
  /// (optinally comparing 
  /// </summary>
  public static class ParamsDump
  {
    /// <summary>
    /// Dump two sets of parameters, optionally showing only differences.
    /// </summary>
    public static void DumpParams(TextWriter writer, bool differentOnly,
                    GameEngineUCISpec externalEngine1Spec, GameEngineUCISpec externalEngine2Spec,
                    NNEvaluatorDef evaluatorDef1, NNEvaluatorDef evaluatorDef2,
                    SearchLimit searchLimit1, SearchLimit searchLimit2,
                    ParamsSelect selectParams1, ParamsSelect selectParams2,
                    ParamsSearch searchParams1, ParamsSearch searchParams2,
                    IManagerGameLimit timeManager1, IManagerGameLimit timeManager2,
                    ParamsSearchExecution paramsSearchExecution1,
                    ParamsSearchExecution paramsSearchExecution2)
    {
      writer.WriteLine("\r\n-----------------------------------------------------------------------");
      writer.WriteLine("ENGINE 1 Options Modifications from Default");

      if (evaluatorDef1 != null)
        writer.WriteLine("Ceres Evaluator : " + evaluatorDef1.ToString());
      else
        writer.Write("External UCI : " + externalEngine1Spec);

      writer.Write(ObjUtils.FieldValuesDumpString<SearchLimit>(searchLimit1, SearchLimit.NodesPerMove(1), differentOnly));
      //      writer.Write(ObjUtils.FieldValuesDumpString<NNEvaluatorDef>(Def.NNEvaluators1.EvaluatorDef, new ParamsNN(), differentOnly));
      writer.Write(ObjUtils.FieldValuesDumpString<ParamsSelect>(selectParams1, new ParamsSelect(), differentOnly));
      writer.Write(ObjUtils.FieldValuesDumpString<ParamsSearch>(searchParams1, new ParamsSearch(), differentOnly));
      DumpTimeManagerDifference(true, null, timeManager1);
      writer.Write(ObjUtils.FieldValuesDumpString<ParamsSearchExecution>(paramsSearchExecution1, new ParamsSearchExecution(), differentOnly));

      writer.WriteLine("\r\n-----------------------------------------------------------------------");
      writer.WriteLine("ENGINE 2 Options Modifications from Engine 1");
      bool evaluatorsDifferent = false;

      bool eval2TypeDifferent = (evaluatorDef1 == null) != (evaluatorDef2 == null);
      if (eval2TypeDifferent)
      {
        evaluatorsDifferent = true;
      }
      else
      {
        if (evaluatorDef1 != null)
          evaluatorsDifferent = evaluatorDef1.ToString() != evaluatorDef2.ToString();
        else
          evaluatorsDifferent = externalEngine1Spec.ToString() != externalEngine2Spec.ToString();
      }
      if (!differentOnly || evaluatorsDifferent)
      {
        if (evaluatorDef1 != null && evaluatorDef2 != null)
          writer.WriteLine("Ceres Evaluator : " + evaluatorDef2.ToString());
        else
          writer.Write("External UCI : " + externalEngine2Spec);


      }

      if (searchLimit2 != null)
      {
        writer.Write(ObjUtils.FieldValuesDumpString<SearchLimit>(searchLimit2, searchLimit1, differentOnly));
        //      writer.Write(ObjUtils.FieldValuesDumpString<NNEvaluatorDef>(Def.NNEvaluators1.EvaluatorDef, Def.NNEvaluators2.EvaluatorDef, differentOnly));
        writer.Write(ObjUtils.FieldValuesDumpString<ParamsSelect>(selectParams2, selectParams1, differentOnly));
        writer.Write(ObjUtils.FieldValuesDumpString<ParamsSearch>(searchParams2, searchParams1, differentOnly));
        DumpTimeManagerDifference(differentOnly, timeManager1, timeManager2);
        writer.Write(ObjUtils.FieldValuesDumpString<ParamsSearchExecution>(paramsSearchExecution2, paramsSearchExecution1, differentOnly));
      }

    }

    public static void DumpTimeManagerDifference(bool differentOnly, IManagerGameLimit timeManager1, IManagerGameLimit timeManager2)
    {
      if (differentOnly)
      {
        if (timeManager1 != timeManager2)
          Console.WriteLine("TimeManager = " + (timeManager2 == null ? "null" : timeManager2.GetType().ToString()));
      }
      else
      {
        Console.WriteLine("TimeManager1 = " + (timeManager1 == null ? "(default)" : timeManager1.GetType().ToString()));
        Console.WriteLine("TimeManager2 = " + (timeManager2 == null ? "(default)" : timeManager2.GetType().ToString()));
      }


    }

  }
}
