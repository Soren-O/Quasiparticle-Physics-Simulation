"""Blank the nameref text field (3rd top-level group) of every \\newlabel
entry in a .aux file, so MiKTeX 2.9's old xr-hyper does not try to expand
document macros (e.g. \\hatg) while importing external labels.
Label numbers, pages, and anchors are untouched."""
import sys

def groups(s, start):
    """Return list of (begin, end) index pairs of top-level {...} groups
    starting at s[start] (which must be '{')."""
    out = []
    i = start
    n = len(s)
    while i < n and s[i] == '{':
        depth = 0
        j = i
        while j < n:
            c = s[j]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            return None  # unbalanced; leave line alone
        out.append((i, j))
        i = j + 1
    return out

def sanitize_line(line):
    tag = '\\newlabel{'
    if not line.startswith(tag):
        return line
    # skip the label-name group, find the payload group
    g = groups(line, len(tag) - 1)  # actually starts at the '{' of the name
    if not g or len(g) < 2:
        return line
    payload_b, payload_e = g[1]
    inner = groups(line, payload_b + 1)
    if not inner or len(inner) < 3:
        return line
    # third top-level subgroup = nameref text; blank it
    b, e = inner[2]
    return line[:b] + '{}' + line[e + 1:]

def main(path):
    with open(path, 'r', encoding='latin-1', newline='') as f:
        lines = f.readlines()
    out = [sanitize_line(l) for l in lines]
    with open(path, 'w', encoding='latin-1', newline='') as f:
        f.writelines(out)
    changed = sum(1 for a, b in zip(lines, out) if a != b)
    print(f"{path}: sanitized {changed} newlabel entries")

if __name__ == '__main__':
    for p in sys.argv[1:]:
        main(p)
