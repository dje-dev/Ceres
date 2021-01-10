#region License notice

/*
Adapted from the PgnFileTools project by Clinton Sheppard 
at https://github.com/handcraftsman/PgnFileTools
licensed under Apache License, Version 2.
*/

#endregion


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
using System.Linq;
using System.Text;
using Ceres.Chess.Textual.PgnFileTools.Extensions;

#endregion

namespace Ceres.Chess.Textual.PgnFileTools
{
  public class GameInfoParser
  {
    private readonly AlgebraicMoveParser _algebraicMoveParser;
    private readonly Stack<StringBuilder> _moveVariations;
    private Func<char, GameInfo, bool> _handle;
    private int _moveNumber;
    private StringBuilder _partial;

    public GameInfoParser()
    {
      _partial = new StringBuilder();
      _algebraicMoveParser = new AlgebraicMoveParser();
      _moveVariations = new Stack<StringBuilder>();
    }

    private static bool Done(char ch, GameInfo gameInfo)
    {
      return true;
    }

    private bool HandleGameComment(char ch, GameInfo gameInfo)
    {
      if (ch == '}')
      {
        gameInfo.Comment = _partial.ToString();
        _partial.Length = 0;
        _handle = HandleMoveNumber;
        return true;
      }
      _partial.Append(ch);
      return true;
    }

    private bool HandleHeaderBody(char ch, GameInfo gameInfo)
    {
      if (ch != ']')
      {
        _partial.Append(ch);
        return true;
      }
      var header = _partial.ToString();
      _partial.Length = 0;
      var quoteLoc = header.IndexOf('"');
      string label;
      string value;
      if (quoteLoc != -1)
      {
        label = header.Substring(0, quoteLoc).TrimEnd();
        value = header.Substring(quoteLoc);
      }
      else
      {
        var headerParts = header.Split(' ');
        label = headerParts[0];
        value = headerParts[1];
      }

      // DJE
      if (!gameInfo.Headers.ContainsKey(label)) gameInfo.Headers.Add(label, value.Trim('"'));
      _handle = HandleHeaderStart;
      return true;
    }

    private bool HandleHeaderStart(char ch, GameInfo gameInfo)
    {
      if (ch == '[')
      {
        _handle = HandleHeaderBody;
        return true;
      }
      if (Char.IsWhiteSpace(ch))
      {
        return true;
      }
      if (ch == '{')
      {
        _partial.Length = 0;
        _handle = HandleGameComment;
        return true;
      }
      if (Char.IsDigit(ch))
      {
        _partial.Length = 0;
        _handle = HandleMoveNumber;
        return HandleMoveNumber(ch, gameInfo);
      }
      return false;
    }

    private bool HandleMoveAnnotation(char ch, GameInfo gameInfo)
    {
      if (Char.IsDigit(ch))
      {
        _partial.Append(ch);
        return true;
      }
      if (_partial.Length > 0)
      {
        gameInfo.Moves.Last().Annotation = Int32.Parse(_partial.ToString());
        _partial.Length = 0;
      }
      _handle = HandleMoveText;
      return HandleMoveText(ch, gameInfo);
    }

    private bool HandleMoveComment(char ch, GameInfo gameInfo)
    {
      if (ch == '}')
      {
        gameInfo.Moves.Last().Comment = _partial.ToString();
        _partial.Length = 0;
        _handle = HandleMoveText;
        return true;
      }
      _partial.Append(ch);
      return true;
    }

    private bool HandleMoveNumber(char ch, GameInfo gameInfo)
    {
      if (Char.IsWhiteSpace(ch))
      {
        return true;
      }
      if (Char.IsDigit(ch))
      {
        _partial.Append(ch);
        return true;
      }
      if (ch == '.')
      {
        if (_partial.Length == 0)
        {
          // Silently consume any extra '.' because they could be from a
          // black first move of game, e.g. "17...Kd5"
          return true;
        }
        else
        {
          _moveNumber = Int32.Parse(_partial.ToString());
          _partial.Length = 0;
          _handle = HandleMoveText;
          return true;
        }
      }
      return false;
    }



    private bool HandleMoveText(char ch, GameInfo gameInfo)
    {
      if (Char.IsWhiteSpace(ch))
      {
        if (_partial.Length == 0)
        {
          return true;
        }

        string partialStr = _partial.ToString();
        bool isResult = partialStr == "*" || partialStr == "0-1" || partialStr == "1-0" || partialStr == "1/2-1/2";
        if (gameInfo.Headers.Count > 0  // this clause added by DJE
            && gameInfo.Headers.ContainsKey("Result")
            && partialStr == gameInfo.Headers["Result"])
        {
          _handle = Done;
          return true;
        }

        var move = _algebraicMoveParser.Parse(partialStr);
        move.Number = _moveNumber;
        gameInfo.Moves.Add(move);
        _partial.Length = 0;
        return true;
      }
      switch (ch)
      {
        case '.':
          return HandleMoveNumber(ch, gameInfo);

        case '{':
          {
            _partial.Length = 0;
            _handle = HandleMoveComment;
            return true;
          }
        case '(':
          {
            _partial.Length = 0;
            _handle = HandleMoveVariation;
            return true;
          }

        case '$':
          {
            _partial.Length = 0;
            _handle = HandleMoveAnnotation;
            return true;
          }
        case '?':
        case '!':
          {
            var move = _algebraicMoveParser.Parse(_partial.ToString());
            move.Number = _moveNumber;
            gameInfo.Moves.Add(move);
            _partial.Length = 0;
            _handle = HandleSymbolicMoveAnnotation;
            return HandleSymbolicMoveAnnotation(ch, gameInfo);
          }
      }

      _partial.Append(ch);
      return true;
    }

    private bool HandleMoveVariation(char ch, GameInfo gameInfo)
    {
      if (ch == '(')
      {
        _moveVariations.Push(_partial);
        _partial = new StringBuilder();
        _partial.Append(ch);
        return true;
      }
      if (ch == ')')
      {
        if (_moveVariations.Count == 0)
        {
          gameInfo.Moves.Last().Variation = _partial.ToString();
          _partial.Length = 0;
          _handle = HandleMoveText;
        }
        else
        {
          var temp = _partial;
          _partial = _moveVariations.Pop();
          _partial.Append(temp);
          _partial.Append(ch);
        }
        return true;
      }
      _partial.Append(ch);
      return true;
    }

    private bool HandleSymbolicMoveAnnotation(char ch, GameInfo gameInfo)
    {
      if (ch == '?' || ch == '!')
      {
        _partial.Append(ch);
        return true;
      }

      gameInfo.Moves.Last().Annotation = SymbolicMoveAnnotation.GetFor(_partial.ToString()).Id;
      _partial.Length = 0;
      _handle = HandleMoveText;
      return HandleMoveText(ch, gameInfo);
    }

    public GameInfo Parse(TextReader source)
    {
      _partial.Length = 0;
      _handle = HandleHeaderStart;
      _moveVariations.Clear();

      var gameInfo = new GameInfo();
      foreach (var ch in source.GenerateFrom())
      {
        var success = _handle(ch, gameInfo);
        if (!success)
        {
          gameInfo.HasError = true;
          break;
        }
        if (_handle == Done)
        {
          break;
        }
      }

      if (!new Func<char, GameInfo, bool>[] { Done }.Contains(_handle))
      {
        gameInfo.HasError = true;
        gameInfo.ErrorMessage = "Unexpected end of game info text.";
      }
      return gameInfo;
    }
  }
}
