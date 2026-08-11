#!/usr/bin/env python3
"""Add dark mode JS wiring to points_html.py."""
path = '/Users/hugo.leclerc/nsdot/applications/reconstruction/viz/points_html.py'
with open(path, 'r') as f:
    lines = f.readlines()

theme_js = [
    '// Dark mode: detect system preference, apply/toggle with d key.\n',
    '(function() {\n',
    'var DARK_ML = window.matchMedia("(prefers-color-scheme: dark)");\n',
    'var curDark = false;\n',
    'function setTheme(on) {\n',
    "  document.body.classList.toggle('dark', on);\n",
    '  if (on) draw(); else draw(); // redraw dots for new color\n',
    '}\n',
    'if (DARK_ML.matches) setTheme(true);\n',
    "DARK_ML.addEventListener('change', e => setTheme(e.matches));\n",
    "window._setTheme = setTheme;\n",  # expose for testing
    "document.addEventListener('keydown', function(dk) {\n",
    "  if (dk.key === 'd' || dk.key === 'D') { dk.preventDefault(); setTheme(!curDark); }\n",
    '});\n',
]

# Find line with updateLabel() near the end and insert theme detection before it
new_lines = []
for i, line in enumerate(lines):
    # Insert after resize() closing, before "// dernière frame par défaut"  
    stripped = line.strip()
    if "resize();" in stripped and not stripped.startswith('//'):
        new_lines.append(line)  # keep resize();
        # Now find the comment about "dernière frame" on next non-empty line(s)  
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines) and 'dernière frame' in lines[j]:
            # Add theme JS before this comment
            new_lines.extend(theme_js)
    new_lines.append(line)

with open(path, 'w') as f:
    f.writelines(new_lines)
print("Done editing!")
