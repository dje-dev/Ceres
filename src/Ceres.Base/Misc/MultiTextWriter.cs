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
using System.IO;
using System.Text;

#endregion

namespace Ceres.Base.Misc
{
  /// <summary>
  /// Subclass of TextWriter that sends output to multiple TextWriters.
  /// </summary>
  public class MultiTextWriter : TextWriter
  {
    private TextWriter[] writers;

    public override Encoding Encoding => throw new NotImplementedException();

    public MultiTextWriter(params TextWriter[] writers) => this.writers = writers;

    public override void Write(char ch)
    {
      foreach (TextWriter writer in writers)
        writer.Write(ch);
    }

    public override void Flush()
    {
      foreach (TextWriter writer in writers)
        writer.Flush();
    }

    public override void Close()
    {
      foreach (TextWriter writer in writers)
        writer.Close();
    }

  }
}
