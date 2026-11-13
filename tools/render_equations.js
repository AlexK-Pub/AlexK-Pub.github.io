// Pre-renders the Wilson-Cowan equations of the TIDE page to static HTML + MathML.
//   npm install katex && node tools/render_equations.js   (run from the repository root)
const katex = require("katex");
const fs = require("fs");
const eqs = [
  "r_E^{(t)} = (1-\\alpha_E)\\,r_E^{(t-1)} + \\alpha_E\\,\\varphi\\big(h_E^{(t)}\\big)",
  "h_E^{(t)} = W_{EE}\\,r_E^{(t-1)} \\textcolor{#B3261E}{-}\\, W_{EI}\\,r_I^{(t-1)} + W_E^{\\mathrm{in}}\\,a^{(t)}",
  "r_I^{(t)} = (1-\\alpha_I)\\,r_I^{(t-1)} + \\alpha_I\\,\\varphi\\big(h_I^{(t)}\\big)",
  "h_I^{(t)} = W_{IE}\\,r_E^{(t-1)} \\textcolor{#B3261E}{-}\\, W_{II}\\,r_I^{(t-1)} + W_I^{\\mathrm{in}}\\,a^{(t)}",
];
// the inhibitory minus sign takes the site's inhibitory color, so it follows the theme
const R = (t) => katex.renderToString(t, { displayMode: false, output: "htmlAndMathml", throwOnError: true }).replace(/color:#B3261E/g, "color:var(--i)");
let out = "{%- comment -%} Pre-rendered with KaTeX " + katex.version + " (tools/render_equations.js). Needs only katex.min.css. {%- endcomment -%}\n<div class=\"eq-grid\">\n";
for (const t of eqs) out += "  <div>" + R(t) + "</div>\n";
out += "</div>\n<p class=\"eq-note\">" + R("\\alpha_\\bullet = \\Delta t/\\tau_\\bullet") + ", &nbsp; " + R("\\tau_E = 20\\,\\mathrm{ms}") + ", &nbsp; " + R("\\tau_I = 5\\,\\mathrm{ms}") + ", &nbsp; and " + R("W_{XY} \\leftarrow \\max(W_{XY},\\,0)") + " after every optimizer step.</p>\n";
fs.writeFileSync("_includes/tide-equations.html", out);
