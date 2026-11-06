import html
from pathlib import Path

def convert_report():
    report_md = Path("REPORT.md").read_text(encoding="utf-8")
    
    css = """
    @media print {
        body { margin: 15mm 20mm; font-size: 11pt; }
        .no-print { display: none; }
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #1f2328;
        max-width: 920px;
        margin: 40px auto;
        padding: 0 25px;
    }
    h1, h2, h3, h4 { color: #0969da; margin-top: 28px; margin-bottom: 14px; font-weight: 600; }
    h1 { font-size: 2.1em; border-bottom: 2px solid #0969da; padding-bottom: 0.3em; }
    h2 { font-size: 1.5em; border-bottom: 1px solid #d0d7de; padding-bottom: 0.3em; margin-top: 36px; }
    h3 { font-size: 1.25em; color: #1f2328; }
    table { border-collapse: collapse; width: 100%; margin: 24px 0; font-size: 0.95em; }
    th, td { border: 1px solid #d0d7de; padding: 10px 14px; text-align: left; }
    th { background-color: #f6f8fa; font-weight: 600; color: #1f2328; }
    tr:nth-child(even) { background-color: #fcfcfc; }
    tr:hover { background-color: #f0f3f6; }
    code {
        background-color: #eff1f3;
        padding: 0.2em 0.4em;
        border-radius: 6px;
        font-size: 88%;
        font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    }
    pre {
        background-color: #f6f8fa;
        padding: 16px;
        overflow: auto;
        border-radius: 6px;
        border: 1px solid #d0d7de;
        line-height: 1.45;
    }
    blockquote {
        margin: 16px 0;
        padding: 8px 16px;
        color: #57606a;
        border-left: 4px solid #0969da;
        background-color: #f6f8fa;
        border-radius: 0 6px 6px 0;
    }
    hr { height: 1px; margin: 30px 0; background-color: #d0d7de; border: 0; }
    img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #d0d7de; margin: 18px 0; }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        background-color: #ddf4ff;
        color: #0969da;
        font-weight: 600;
        font-size: 0.85em;
    }
    """

    lines = report_md.splitlines()
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        "<title>FEVER Information Retrieval - Technical Report</title>",
        f"<style>{css}</style>",
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">',
        '<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>',
        '<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body);"></script>',
        "</head>",
        "<body>",
    ]

    in_table = False
    in_code = False
    in_list = False

    for line in lines:
        if line.startswith("```"):
            if in_code:
                html_lines.append("</code></pre>")
                in_code = False
            else:
                html_lines.append("<pre><code>")
                in_code = True
            continue
        if in_code:
            html_lines.append(html.escape(line))
            continue

        if line.startswith("|") and "|" in line[1:]:
            if "---" in line:
                continue
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if not in_table:
                in_table = True
                html_lines.append("<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr></thead><tbody>")
            else:
                html_lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            continue
        elif in_table:
            html_lines.append("</tbody></table>")
            in_table = False

        if line.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{line[2:]}</li>")
            continue
        elif in_list and not line.startswith("- "):
            html_lines.append("</ul>")
            in_list = False

        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("#### "):
            html_lines.append(f"<h4>{line[5:]}</h4>")
        elif line.startswith("---"):
            html_lines.append("<hr>")
        elif line.startswith("> "):
            html_lines.append(f"<blockquote>{line[2:]}</blockquote>")
        elif "artifacts/precision_recall.png" in line:
            html_lines.append('<img src="artifacts/precision_recall.png" alt="Precision-Recall Graph">')
        elif line.strip():
            html_lines.append(f"<p>{line}</p>")

    if in_table:
        html_lines.append("</tbody></table>")
    if in_list:
        html_lines.append("</ul>")
    if in_code:
        html_lines.append("</code></pre>")

    html_lines.append("</body></html>")

    output_path = Path("REPORT.html")
    output_path.write_text("\n".join(html_lines), encoding="utf-8")
    print(f"Exported styled report to {output_path.resolve()}")

if __name__ == "__main__":
    convert_report()

