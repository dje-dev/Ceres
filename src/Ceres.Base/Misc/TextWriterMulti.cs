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
using System.Text;

#endregion

namespace Ceres.Base.Misc
{
  /// <summary>
  /// Subclass of TextWriter which replicates output to possibly multiple TextWriters.
  /// 
  /// Thanks to WaffleSouffle (https://stackoverflow.com/questions/8823562/how-to-split-a-single-textwriter-into-multiple-outputs-in-net).
  /// </summary>
  public class TextWriterMulti : TextWriter
  {
    private List<TextWriter> writers = new List<TextWriter>();
    private IFormatProvider formatProvider = null;
    private Encoding encoding = null;

    public bool PrefixDateTime { get; set; } = false;

    #region TextWriter Properties
    public override IFormatProvider FormatProvider
    {
      get
      {
        IFormatProvider formatProvider = this.formatProvider;
        if (formatProvider == null)
        {
          formatProvider = base.FormatProvider;
        }
        return formatProvider;
      }
    }

    public override string NewLine
    {
      get { return base.NewLine; }

      set
      {
        foreach (TextWriter writer in this.writers)
        {
          writer.NewLine = value;
        }

        base.NewLine = value;
      }
    }


    public override Encoding Encoding
    {
      get
      {
        Encoding encoding = this.encoding;

        if (encoding == null)
        {
          encoding = Encoding.Default;
        }

        return encoding;
      }
    }

    #region TextWriter Property Setters

    TextWriterMulti SetFormatProvider(IFormatProvider value)
    {
      this.formatProvider = value;
      return this;
    }

    TextWriterMulti SetEncoding(Encoding value)
    {
      this.encoding = value;
      return this;
    }
    #endregion // TextWriter Property Setters
    #endregion // TextWriter Properties


    #region Construction/Destruction
    public TextWriterMulti(params TextWriter[] writers)
    {
      this.Clear();
      this.AddWriters(writers);
    }
    #endregion // Construction/Destruction

    #region Public interface
    public TextWriterMulti Clear()
    {
      this.writers.Clear();
      return this;
    }

    public TextWriterMulti AddWriter(TextWriter writer)
    {
      this.writers.Add(writer);
      return this;
    }

    public TextWriterMulti AddWriters(IEnumerable<TextWriter> writers)
    {
      this.writers.AddRange(writers);
      return this;
    }
    #endregion // Public interface

    #region TextWriter methods

    public override void Close()
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.Close();
      }
      base.Close();
    }

    protected override void Dispose(bool disposing)
    {
      foreach (TextWriter writer in this.writers)
      {
        if (disposing)
        {
          writer.Dispose();
        }
      }
      base.Dispose(disposing);
    }

    public override void Flush()
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.Flush();
      }

      base.Flush();
    }


    public override void Write(bool value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.Write(value);
      }
    }

    public override void Write(char value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.Write(value);
      }
    }

    public override void Write(char[] buffer)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.Write(buffer);
      }
    }

    public override void Write(decimal value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.Write(value);
      }
    }

    public override void Write(double value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.Write(value);
      }
    }

    public override void Write(float value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.Write(value);
      }
    }

    public override void Write(int value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.Write(value);
      }
    }

    public override void Write(long value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.Write(value);
      }
    }

    public override void Write(object value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.Write(value);
      }
    }

    public override void Write(string value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.Write(value);
      }
    }

    public override void Write(uint value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.Write(value);
      }
    }

    public override void Write(ulong value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.Write(value);
      }
    }

    public override void Write(string format, object arg0)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.Write(format, arg0);
      }

    }

    public override void Write(string format, params object[] arg)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.Write(format, arg);
      }
    }

    public override void Write(char[] buffer, int index, int count)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.Write(buffer, index, count);
      }
    }

    public override void Write(string format, object arg0, object arg1)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.Write(format, arg0, arg1);
      }
    }

    public override void Write(string format, object arg0, object arg1, object arg2)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.Write(format, arg0, arg1, arg2);
      }
    }

    public override void WriteLine()
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.WriteLine();
      }
    }

    public override void WriteLine(bool value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.WriteLine(value);
      }
    }

    public override void WriteLine(char value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.WriteLine(value);
      }
    }

    public override void WriteLine(char[] buffer)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.WriteLine(buffer);
      }
    }

    public override void WriteLine(decimal value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.WriteLine(value);
      }
    }

    public override void WriteLine(double value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.WriteLine(value);
      }
    }

    public override void WriteLine(float value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.WriteLine(value);
      }
    }

    public override void WriteLine(int value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.WriteLine(value);
      }
    }

    public override void WriteLine(long value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.WriteLine(value);
      }
    }

    public override void WriteLine(object value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.WriteLine(value);
      }
    }

    public override void WriteLine(string value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.WriteLine(value);
      }
    }

    public override void WriteLine(uint value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.WriteLine(value);
      }
    }

    public override void WriteLine(ulong value)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.WriteLine(value);
      }
    }

    public override void WriteLine(string format, object arg0)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.WriteLine(format, arg0);
      }
    }

    public override void WriteLine(string format, params object[] arg)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.WriteLine(format, arg);
      }
    }

    public override void WriteLine(char[] buffer, int index, int count)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.WriteLine(buffer, index, count);
      }
    }

    public override void WriteLine(string format, object arg0, object arg1)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.WriteLine(format, arg0, arg1);
      }
    }

    public override void WriteLine(string format, object arg0, object arg1, object arg2)
    {
      foreach (TextWriter writer in this.writers)
      {
        writer.WriteLine(format, arg0, arg1, arg2);
      }
    }
    #endregion // TextWriter methods
  }
}
