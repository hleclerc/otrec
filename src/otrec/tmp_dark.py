#!/usr/bin/env python3 -u
"""Apply dark mode edits to points_html.py."""
path = '/Users/hugo.leclerc/nsdot/applications/reconstruction/viz/points_html.py'
with open(path, 'r') as f:
    lines = f.readlines()

# ============================================================
# Edit 1: Make ctx.fillStyle theme-aware (line 175)
# ============================================================
for i in range(len(lines)):
     if "ctx.fillStyle = '#000000';" in lines[i]:
          lines[i] = lines[i].replace(
              "#000000",
              "DARK ? '#ffffff' : '#000000'",
              1,
          )
          break

# ============================================================
# Edit 2: Add theme detection + toggle (before resize() call)
# ============================================================
for i in range(len(lines)):
     s = lines[i]
     if "resize();" in s and not s.startswith("//"):
         # This is the resize() call line. Insert theme code BEFORE it.
         idx = lines.index(s)  # actual index
          # Find this exact line by checking content
         break

# Simpler: just find the FIRST occurrence of "resize();" as an isolated line
for i in range(len(lines)):
     if lines[i].strip() == "resize();":
        insertion_idx = i
         break
else:
    raise ValueError("resize() not found")

theme_code = [
    "// Dark mode: detect system preference, toggle with d.\n",
    "(function() {\n",
     "var darkML = window.matchMedia('(prefers-color-scheme: dark)');\n",
     "var curDark = false;\n",
     "function setTheme(on) {\n",
      "curDark = on;\n",
       "document.body.classList.toggle('dark', on);\n",
        "updateFillColors();\n",  # redraw so dots change color
         "draw();\n",
        "}\n",
         "if (darkML.matches) setTheme(true);\n",
          "darkML.addEventListener('change', e => setTheme(e.matches));\n",
           "var darkHandler = function(ev) {\n",
             "if (ev.key === 'd' || ev.key === 'D') {\n",
               "ev.preventDefault();\n",
                "setTheme(!curDark);\n",
                 "}\n",
                  "};\n",
                   "document.addEventListener('keydown', darkHandler);\n",
                    "})();\n",
]

lines.insert(insertion_idx, "\n".join(theme_code))

with open(path, 'w') as f:
    f.writelines(lines)
print("Done!")

