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
using System.Text.RegularExpressions;

#endregion

namespace Ceres.MCGS.GameEngines;

/// <summary>
/// Formats a raw diagnostic "minilog" text file (as produced by <see cref="MCGSMiniLog"/>)
/// into a single, self-contained HTML page (inline CSS, no external dependencies) suitable for
/// browsing: a collapsible configuration header, a per-game table of move diagnostics, an
/// expandable candidate-move list per move, and a per-game result footer.
/// </summary>
public static class MCGSMiniLogHtmlFormatter
{
  /// <summary>
  /// Reads the given raw minilog file and returns a complete standalone HTML document as a string.
  /// </summary>
  public static string FormatToHtml(string logFilePath)
  {
    if (logFilePath == null)
    {
      throw new ArgumentNullException(nameof(logFilePath));
    }

    string[] lines = ReadAllLinesShared(logFilePath);
    return FormatLinesToHtml(lines, Path.GetFileName(logFilePath));
  }


  /// <summary>
  /// Reads the given raw minilog file and writes a complete standalone HTML document to the
  /// specified output path.
  /// </summary>
  public static void WriteHtmlFile(string logFilePath, string htmlOutputPath)
  {
    File.WriteAllText(htmlOutputPath, FormatToHtml(logFilePath));
  }


  // Scalar columns rendered for each move, in display order (label -> token key on the move line).
  static readonly (string Label, string Key)[] ScalarColumns =
  {
    ("RootN", "RootN"),
    ("StoreN", "StoreN"),
    ("NNEvals", "NNEvals"),
    ("NFirst%", "NFirst%"),
    ("TimeRem", "TimeRem"),
    ("OppTimeRem", "OppTimeRem"),
    ("LimInit", "LimInit"),
    ("Budget%", "BudgetFrac"),
    ("NPS", "NPS"),
    ("EPS", "EPS"),
    ("Backend", "BackendBusy"),
    ("Depth", "Depth"),
    ("SelD", "SelDepth"),
  };

  // Scalar columns whose displayed value is reformatted with thousands separators.
  static readonly System.Collections.Generic.HashSet<string> ThousandsKeys = new() { "RootN", "StoreN" };

  static readonly Regex CandidateRegex = new Regex(@"\*?\([^()]*\)", RegexOptions.Compiled);


  static string FormatLinesToHtml(string[] lines, string title)
  {
    int firstGameIdx = lines.Length;
    for (int i = 0; i < lines.Length; i++)
    {
      if (lines[i].StartsWith("=== NEW GAME:"))
      {
        firstGameIdx = i;
        break;
      }
    }

    StringBuilder header = new StringBuilder();
    for (int i = 0; i < firstGameIdx; i++)
    {
      header.AppendLine(lines[i]);
    }

    StringBuilder sb = new StringBuilder();
    sb.AppendLine("<!DOCTYPE html>");
    sb.AppendLine("<html lang=\"en\">");
    sb.AppendLine("<head>");
    sb.AppendLine("<meta charset=\"utf-8\">");
    sb.AppendLine("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">");
    sb.AppendLine("<title>" + Esc(title) + "</title>");
    sb.AppendLine(StyleBlock());
    sb.AppendLine("</head>");
    sb.AppendLine("<body>");

    sb.AppendLine("<h1>Ceres MCGS Minilog</h1>");
    sb.AppendLine("<div class=\"sub\">" + Esc(title) + "</div>");

    WriteHeaderSection(sb, header.ToString());

    int gameNumber = 0;
    int idx = firstGameIdx;
    while (idx < lines.Length)
    {
      if (!lines[idx].StartsWith("=== NEW GAME:"))
      {
        idx++;
        continue;
      }

      string gameID = ExtractGameID(lines[idx]);
      idx++;

      List<string> body = new List<string>();
      while (idx < lines.Length && !lines[idx].StartsWith("=== NEW GAME:"))
      {
        body.Add(lines[idx]);
        idx++;
      }

      gameNumber++;
      WriteGameSection(sb, gameNumber, gameID, body);
    }

    if (gameNumber == 0)
    {
      sb.AppendLine("<p class=\"note\">(No games found in this log.)</p>");
    }

    sb.AppendLine("</body>");
    sb.AppendLine("</html>");
    return sb.ToString();
  }


  static void WriteHeaderSection(StringBuilder sb, string headerText)
  {
    // Pull a few key fields out for a prominent summary card.
    string engine = FindHeaderValue(headerText, "Engine");
    string evaluator = FindHeaderValue(headerText, "Evaluator");
    string limit = FindHeaderValue(headerText, "AssignedSearchLimit");
    string timestamp = FindHeaderValue(headerText, "Timestamp");
    string machine = FindHeaderValue(headerText, "Machine");

    sb.AppendLine("<div class=\"card\">");
    sb.AppendLine("<table class=\"kv\">");
    AppendKV(sb, "Engine", engine);
    AppendKV(sb, "Evaluator", evaluator);
    AppendKV(sb, "Assigned search limit", limit);
    AppendKV(sb, "Timestamp", timestamp);
    AppendKV(sb, "Machine", machine);
    sb.AppendLine("</table>");
    sb.AppendLine("</div>");

    sb.AppendLine("<details class=\"hdr\">");
    sb.AppendLine("<summary>Full configuration header (system info + ParamsSearch / ParamsSelect dumps)</summary>");
    sb.AppendLine("<pre>" + Esc(headerText) + "</pre>");
    sb.AppendLine("</details>");
  }


  static void WriteGameSection(StringBuilder sb, int gameNumber, string gameID, List<string> body)
  {
    // Parse the game body in order into rows (move lines and inline blunder / limits blocks) plus
    // the footer. Row Kind is one of "move", "blunder", "limits".
    List<(string Kind, string Text)> rows = new List<(string Kind, string Text)>();
    List<string> footerLines = new List<string>();
    int moveCount = 0;

    bool inFooter = false;
    bool inBlunder = false;
    bool inLimits = false;
    List<string> blockLines = null;
    foreach (string l in body)
    {
      if (inFooter)
      {
        footerLines.Add(l);
        if (l.StartsWith("=== END GAME RESULT ==="))
        {
          inFooter = false;
        }
        continue;
      }

      if (inBlunder)
      {
        if (l.StartsWith("=== END BLUNDER ==="))
        {
          rows.Add(("blunder", string.Join("\n", blockLines)));
          inBlunder = false;
          blockLines = null;
        }
        else
        {
          blockLines.Add(l);
        }
        continue;
      }

      if (inLimits)
      {
        if (l.StartsWith("=== END LIMITS ==="))
        {
          rows.Add(("limits", string.Join("\n", blockLines)));
          inLimits = false;
          blockLines = null;
        }
        else
        {
          blockLines.Add(l);
        }
        continue;
      }

      if (l.StartsWith("=== GAME RESULT ==="))
      {
        inFooter = true;
        footerLines.Add(l);
      }
      else if (l.StartsWith("=== BLUNDER ==="))
      {
        inBlunder = true;
        blockLines = new List<string>();
      }
      else if (l.StartsWith("=== LIMITS ==="))
      {
        inLimits = true;
        blockLines = new List<string>();
      }
      else if (l.StartsWith("FEN="))
      {
        rows.Add(("move", l));
        moveCount++;
      }
    }

    sb.AppendLine("<h2>Game " + gameNumber + " <span class=\"gid\">" + Esc(gameID) + "</span></h2>");

    string footerText = string.Join("\n", footerLines);
    string result = FindFooterValue(footerText, "Result");
    if (result.Length > 0)
    {
      // Per-engine total time spent and time remaining (this engine vs opponent), if recorded.
      string thisID = FindFooterValue(footerText, "ThisEngine");
      string oppID = FindFooterValue(footerText, "Opponent");
      string thisTotal = FindFooterValue(footerText, "ThisTotalTime");
      string thisRem = FindFooterValue(footerText, "ThisRemainingTime");
      string oppTotal = FindFooterValue(footerText, "OppTotalTime");
      string oppRem = FindFooterValue(footerText, "OppRemainingTime");

      string timeSummary = "";
      if (thisTotal.Length > 0 || oppTotal.Length > 0)
      {
        timeSummary = " &nbsp;&mdash;&nbsp; <span class=\"timesummary\">"
                    + Esc(thisID.Length > 0 ? thisID : "Ceres") + ": " + FormatFooterSeconds(thisTotal)
                    + " used, " + FormatFooterSeconds(thisRem) + " left &nbsp;&middot;&nbsp; "
                    + Esc(oppID.Length > 0 ? oppID : "opponent") + ": " + FormatFooterSeconds(oppTotal)
                    + " used, " + FormatFooterSeconds(oppRem) + " left</span>";
      }

      sb.AppendLine("<div class=\"resultline\">Result: <span class=\"" + ResultClass(result) + "\">"
                    + Esc(result) + "</span> &nbsp; (" + moveCount + " moves logged)" + timeSummary + "</div>");
    }

    WriteMovesTable(sb, rows);

    if (footerLines.Count > 0)
    {
      sb.AppendLine("<details class=\"footer\">");
      sb.AppendLine("<summary>Game result detail</summary>");
      sb.AppendLine("<pre>" + Esc(footerText) + "</pre>");
      sb.AppendLine("</details>");
    }
  }


  static void WriteMovesTable(StringBuilder sb, List<(string Kind, string Text)> rows)
  {
    if (rows.Count == 0)
    {
      sb.AppendLine("<p class=\"note\">(No move lines.)</p>");
      return;
    }

    int totalCols = 1 + ScalarColumns.Length + 3;

    sb.AppendLine("<div class=\"tablewrap\">");
    sb.AppendLine("<table class=\"moves\">");
    sb.Append("<thead><tr><th>#</th>");
    foreach ((string label, string key) in ScalarColumns)
    {
      sb.Append("<th>" + Esc(label) + "</th>");
    }
    sb.Append("<th>Played</th><th>Candidates</th><th>FEN</th></tr></thead>");
    sb.AppendLine();
    sb.AppendLine("<tbody>");

    // RootN of the first logged move in the game, used to compute NFirst% (RootN relative to move 1).
    long firstRootN = -1;
    int ply = 0;
    foreach ((string kind, string text) in rows)
    {
      if (kind == "blunder")
      {
        // Blunder diagnostics: a plain fixed-font text block spanning the full table width.
        sb.Append("<tr class=\"blunderrow\"><td colspan=\"" + totalCols + "\">");
        sb.Append("<details><summary class=\"blundersummary\">BLUNDER diagnostics (click to expand)</summary>");
        sb.Append("<pre class=\"blunder\">" + Esc(text) + "</pre></details>");
        sb.AppendLine("</td></tr>");
        continue;
      }

      if (kind == "limits")
      {
        // Limits-manager allocation reasoning for the move that follows: a fixed-font text block
        // spanning the full table width (collapsed by default to keep the table compact).
        sb.Append("<tr class=\"limitsrow\"><td colspan=\"" + totalCols + "\">");
        sb.Append("<details><summary class=\"limitssummary\">LIMITS allocation (click to expand)</summary>");
        sb.Append("<pre class=\"limits\">" + Esc(text) + "</pre></details>");
        sb.AppendLine("</td></tr>");
        continue;
      }

      ply++;
      ParsedMove m = ParseMoveLine(text);

      // Capture the first move's (positive) RootN as the baseline for NFirst%.
      if (firstRootN < 0 && m.Scalars.TryGetValue("RootN", out string firstRootNStr)
          && long.TryParse(firstRootNStr, out long parsedFirstRootN) && parsedFirstRootN > 0)
      {
        firstRootN = parsedFirstRootN;
      }

      sb.Append("<tr>");
      sb.Append("<td class=\"num\">" + ply + "</td>");

      foreach ((string label, string key) in ScalarColumns)
      {
        string val = FormatScalarCell(key, m, firstRootN);
        string cls = key == "BudgetFrac" ? "num budget" : "num";
        sb.Append("<td class=\"" + cls + "\">" + Esc(val) + "</td>");
      }

      // The candidate list is sorted by visits descending, so the top-N (most-visited) move is the
      // first entry. The played move differs from top-N (for any reason) when it is not that entry.
      bool playedIsTopN = m.Candidates.Count > 0 && m.Candidates[0].Played;
      bool anyPlayed = false;
      foreach (ParsedCandidate c in m.Candidates)
      {
        if (c.Played) { anyPlayed = true; break; }
      }
      bool playedOverridesTopN = anyPlayed && !playedIsTopN;

      // Played move (with Q). A leading '*' marks a played move that was not the top-N move.
      string playedCell = "-";
      foreach (ParsedCandidate c in m.Candidates)
      {
        if (c.Played)
        {
          string star = playedOverridesTopN ? "<span class=\"override\">*</span>" : "";
          playedCell = star + "<span class=\"playedmove\">" + Esc(c.San) + "</span> <span class=\"q\">"
                       + Esc(c.Q) + "</span>";
          break;
        }
      }
      sb.Append("<td class=\"played\">" + playedCell + "</td>");

      // Candidate list (collapsible), preceded by a brief reasoning line when the played move was not
      // the top-N move - naming the mechanism if recorded (best-Q / minimax / irreversible / drp-avoid).
      sb.Append("<td class=\"cands\">");
      if (m.Candidates.Count > 0)
      {
        sb.Append("<details><summary>" + m.Candidates.Count + " moves</summary><div class=\"candlist\">");
        if (playedOverridesTopN)
        {
          string mech = m.Scalars.TryGetValue("Sel", out string selVal) && selVal.Length > 0 ? selVal : "—";
          ParsedCandidate top = m.Candidates[0];
          sb.Append("<div class=\"reason\">played ≠ top-N · <b>" + Esc(mech)
                    + "</b> · top-N " + Esc(top.San) + " (" + Esc(top.Visit)
                    + ", Q " + Esc(top.Q) + ")</div>");
        }
        foreach (ParsedCandidate c in m.Candidates)
        {
          string badgeCls = c.Played ? "cand played" : "cand";
          sb.Append("<span class=\"" + badgeCls + "\"><b>" + Esc(c.San) + "</b> "
                    + Esc(c.Visit) + " <span class=\"q\">" + Esc(c.Q) + "</span></span>");
        }
        sb.Append("</div></details>");
      }
      sb.Append("</td>");

      sb.Append("<td class=\"fen\">" + Esc(m.Fen) + "</td>");
      sb.AppendLine("</tr>");
    }

    sb.AppendLine("</tbody></table></div>");
  }


  #region Parsing

  class ParsedMove
  {
    public string Fen = "";
    public string SideToMove = "";
    public Dictionary<string, string> Scalars = new Dictionary<string, string>();
    public List<ParsedCandidate> Candidates = new List<ParsedCandidate>();
  }

  class ParsedCandidate
  {
    public string San = "";
    public string Visit = "";
    public string Q = "";
    public bool Played;
  }


  // Produces the display string for one scalar column cell, applying per-column formatting:
  // NFirst% is derived (RootN relative to the first move's RootN), node columns get thousands
  // separators, and Depth is rounded to the nearest integer. Other columns pass through verbatim.
  static string FormatScalarCell(string key, ParsedMove m, long firstRootN)
  {
    System.Globalization.CultureInfo ci = System.Globalization.CultureInfo.InvariantCulture;

    if (key == "NFirst%")
    {
      if (firstRootN > 0 && m.Scalars.TryGetValue("RootN", out string rootNStr)
          && long.TryParse(rootNStr, out long rootN))
      {
        long pct = (long)Math.Round(100.0 * rootN / firstRootN, MidpointRounding.AwayFromZero);
        return pct.ToString(ci) + "%";
      }
      return "";
    }

    string val = m.Scalars.TryGetValue(key, out string v) ? v : "";
    if (val.Length == 0)
    {
      return "";
    }

    if (ThousandsKeys.Contains(key) && long.TryParse(val, out long n))
    {
      return n.ToString("N0", ci);
    }

    if (key == "Depth" && double.TryParse(val, System.Globalization.NumberStyles.Float, ci, out double d))
    {
      return ((long)Math.Round(d, MidpointRounding.AwayFromZero)).ToString(ci);
    }

    return val;
  }


  static ParsedMove ParseMoveLine(string line)
  {
    ParsedMove m = new ParsedMove();

    int bar = line.IndexOf(" | ", StringComparison.Ordinal);
    string scalarPart = bar >= 0 ? line.Substring(0, bar) : line;
    string candPart = bar >= 0 ? line.Substring(bar + 3) : "";

    string rest = scalarPart;
    if (scalarPart.StartsWith("FEN=\""))
    {
      int endQuote = scalarPart.IndexOf('"', 5);
      if (endQuote > 0)
      {
        m.Fen = scalarPart.Substring(5, endQuote - 5);
        rest = scalarPart.Substring(endQuote + 1);
        if (rest.StartsWith(","))
        {
          rest = rest.Substring(1);
        }
        rest = rest.Trim();
      }
    }

    foreach (string tok in rest.Split(", ", StringSplitOptions.RemoveEmptyEntries))
    {
      int eq = tok.IndexOf('=');
      if (eq > 0)
      {
        m.Scalars[tok.Substring(0, eq)] = tok.Substring(eq + 1).Trim();
      }
    }

    string[] fenParts = m.Fen.Split(' ');
    if (fenParts.Length >= 2)
    {
      m.SideToMove = fenParts[1] == "w" ? "White" : fenParts[1] == "b" ? "Black" : fenParts[1];
    }

    foreach (Match match in CandidateRegex.Matches(candPart))
    {
      string v = match.Value;
      ParsedCandidate c = new ParsedCandidate();
      c.Played = v.StartsWith("*");
      int op = v.IndexOf('(');
      int cp = v.LastIndexOf(')');
      if (op >= 0 && cp > op)
      {
        string inner = v.Substring(op + 1, cp - op - 1);
        string[] parts = inner.Split(", ");
        c.San = parts.Length > 0 ? parts[0] : "";
        c.Visit = parts.Length > 1 ? parts[1] : "";
        c.Q = parts.Length > 2 ? parts[2] : "";
      }
      m.Candidates.Add(c);
    }

    return m;
  }


  static string ExtractGameID(string newGameLine)
  {
    // Format: "=== NEW GAME: <id> ==="
    string s = newGameLine.Replace("=== NEW GAME:", "").Trim();
    if (s.EndsWith("==="))
    {
      s = s.Substring(0, s.Length - 3).Trim();
    }
    return s;
  }


  static string FindHeaderValue(string headerText, string keyPrefix)
  {
    foreach (string line in headerText.Split('\n'))
    {
      string t = line.TrimEnd('\r');
      if (t.TrimStart().StartsWith(keyPrefix))
      {
        int colon = t.IndexOf(':');
        if (colon >= 0)
        {
          return t.Substring(colon + 1).Trim();
        }
      }
    }
    return "";
  }


  // Formats a raw footer seconds value (e.g. "161.2334") as "161.23s"; returns "n/a" when absent
  // and the raw text when unparseable.
  static string FormatFooterSeconds(string raw)
  {
    if (string.IsNullOrEmpty(raw))
    {
      return "n/a";
    }
    if (double.TryParse(raw, System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture, out double secs))
    {
      return secs.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "s";
    }
    return raw;
  }


  static string FindFooterValue(string footerText, string key)
  {
    // Footer tokens look like "Key=Value" possibly several per line (space separated).
    foreach (string line in footerText.Split('\n'))
    {
      foreach (string tok in line.Split(' '))
      {
        int eq = tok.IndexOf('=');
        if (eq > 0 && tok.Substring(0, eq) == key)
        {
          return tok.Substring(eq + 1).Trim();
        }
      }
    }
    return "";
  }

  #endregion


  static void AppendKV(StringBuilder sb, string key, string value)
  {
    sb.AppendLine("<tr><th>" + Esc(key) + "</th><td>" + Esc(value) + "</td></tr>");
  }


  static string ResultClass(string result)
  {
    if (result == "Win")
    {
      return "win";
    }
    if (result == "Loss")
    {
      return "loss";
    }
    if (result == "Draw")
    {
      return "draw";
    }
    return "other";
  }


  // Reads all lines allowing shared read/write access, so the log can be rendered to HTML even
  // while the producing MCGSMiniLog still holds the file open for writing (required on Windows).
  static string[] ReadAllLinesShared(string path)
  {
    List<string> lines = new List<string>();
    using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
    using (StreamReader sr = new StreamReader(fs))
    {
      string line;
      while ((line = sr.ReadLine()) != null)
      {
        lines.Add(line);
      }
    }
    return lines.ToArray();
  }


  static string Esc(string s)
  {
    if (string.IsNullOrEmpty(s))
    {
      return "";
    }
    return s.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("\"", "&quot;");
  }


  static string StyleBlock()
  {
    return @"<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#2a2f3a; --txt:#d6d9df; --muted:#8a909c;
          --accent:#4ea1ff; --played:#ffd24a; --win:#36c275; --loss:#ff5d5d; --draw:#9aa0ac; }
  * { box-sizing: border-box; }
  body { margin: 18px; background: var(--bg); color: var(--txt);
         font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; font-size: 15px; }
  h1 { font-size: 23px; margin: 0 0 2px 0; }
  h2 { font-size: 19px; margin: 26px 0 8px 0; border-bottom: 1px solid var(--line); padding-bottom: 4px; }
  .sub { color: var(--muted); margin-bottom: 14px; word-break: break-all; }
  .gid { color: var(--muted); font-weight: normal; font-size: 14px; }
  .note { color: var(--muted); }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
          padding: 8px 12px; display: inline-block; margin-bottom: 10px; }
  table.kv { border-collapse: collapse; }
  table.kv th { text-align: left; color: var(--muted); font-weight: 600; padding: 2px 14px 2px 0; vertical-align: top; }
  table.kv td { padding: 2px 0; font-family: ui-monospace, Consolas, monospace; }
  details.hdr, details.footer { margin: 6px 0 12px 0; }
  details.hdr > summary, details.footer > summary { cursor: pointer; color: var(--accent); margin-bottom: 6px; }
  pre { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px;
        overflow-x: auto; font-family: ui-monospace, Consolas, monospace; font-size: 13.5px; line-height: 1.4; }
  .resultline { margin: 4px 0 8px 0; }
  .resultline .timesummary { color: var(--muted); font-family: ui-monospace, Consolas, monospace; font-size: 13px; }
  .resultline .win { color: var(--win); font-weight: 700; }
  .resultline .loss { color: var(--loss); font-weight: 700; }
  .resultline .draw { color: var(--draw); font-weight: 700; }
  .resultline .other { color: var(--accent); font-weight: 700; }
  .tablewrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }
  table.moves { border-collapse: collapse; width: 100%; font-size: 14px; }
  table.moves thead th { position: sticky; top: 0; background: #1d2230; color: var(--txt);
                         text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--line); white-space: nowrap; }
  table.moves thead th:nth-child(1) { text-align: left; }
  table.moves td { padding: 4px 8px; border-bottom: 1px solid #20242e; vertical-align: top; }
  table.moves tbody tr:nth-child(odd) { background: #13161d; }
  td.num { text-align: right; font-family: ui-monospace, Consolas, monospace; white-space: nowrap; }
  td.budget { color: var(--accent); }
  td.played .playedmove { color: var(--played); font-weight: 700; }
  td.played .override { color: var(--loss); font-weight: 700; margin-right: 2px; }
  .candlist .reason { width: 100%; color: var(--muted); font-size: 12px; margin-bottom: 4px; }
  .candlist .reason b { color: var(--accent); }
  .q { color: var(--muted); }
  td.fen { font-family: ui-monospace, Consolas, monospace; font-size: 13px; color: var(--muted); white-space: nowrap; }
  td.cands details > summary { cursor: pointer; color: var(--accent); white-space: nowrap; }
  .candlist { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; max-width: 720px; }
  .cand { border: 1px solid var(--line); border-radius: 5px; padding: 1px 6px; background: #12151c; white-space: nowrap; font-size: 13px; }
  .cand.played { border-color: var(--played); background: #2a2510; }
  .cand b { font-weight: 600; }
  tr.blunderrow td { background: #2a1414; }
  .blundersummary { color: var(--loss); font-weight: 700; cursor: pointer; }
  pre.blunder { background: #140d0d; border-color: #5a2a2a; font-family: Consolas, ui-monospace, monospace;
                font-size: 12px; margin: 6px 0 2px 0; }
  tr.limitsrow td { background: #101a26; }
  .limitssummary { color: var(--accent); font-weight: 700; cursor: pointer; }
  pre.limits { background: #0c141d; border-color: #244055; font-family: Consolas, ui-monospace, monospace;
               font-size: 12px; margin: 6px 0 2px 0; }
</style>";
  }
}
