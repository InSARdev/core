# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://InSAR.dev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------
"""
Writes vmf3_grid.txt and gmf.txt of this folder from the TU Wien sources vmf3_grid.m and gmf.f.

vmf3_grid.m: the matrices 'name = [ ... ];' (rows separated by ';', decimal strings kept as written).
gmf.f: the DATA statements 'data (name(i),i=1,55)/ ... /' (values written with %e).
Usage: python3.13 mapping_tables.py vmf3_grid.m gmf.f [outdir]   (outdir defaults to this folder)
Sources and their terms: README.md and LICENSE in this folder. insardev_toolkit does not import this file.
"""
import os
import re
import sys
import hashlib

VMF3_NAMES = ['anm_bh', 'bnm_bh', 'anm_bw', 'bnm_bw', 'anm_ch', 'bnm_ch', 'anm_cw', 'bnm_cw']
GMF_NAMES = ['ah_mean', 'bh_mean', 'ah_amp', 'bh_amp', 'aw_mean', 'bw_mean', 'aw_amp', 'bw_amp']
VMF3_HEADER = ['# VMF3 spherical harmonic coefficients of vmf3_grid.m (TU Wien): 8 tables of 91 rows of 5 values.', '# Source: https://vmf.geo.tuwien.ac.at/codes/vmf3_grid.m (Landskron and Boehm 2017, doi:10.1007/s00190-017-1066-2).', '# Layout: a line [name] starts a table; the rows that follow hold the 5 values A0 A1 B1 A2 B2 of each (n, m) pair', '# (n up to 12) in the order of vmf3_grid.m, as the decimal strings of the source literal. Blank and # lines are skipped.']
GMF_HEADER = ['# GMF coefficients of gmf.f (TU Wien): 8 tables of 55 values, the terms of degree 0 to 9 in utils_tropo order.', '# Source: https://vmf.geo.tuwien.ac.at/codes/gmf.f (Boehm, Niell, Tregoning, Schuh 2006, doi:10.1029/2005GL025545).', '# Layout: a line [name] starts a table; its 11 rows hold 5 values each, written as %e (6 decimals).', '# Blank and # lines are skipped.']
NUMBER = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?')


def vmf3_tables(text):
    """Return {name: 91 row strings of 5 values} of the matrices of vmf3_grid.m."""
    tables = {}
    for name in VMF3_NAMES:
        body = re.search(r'\b' + name + r'\s*=\s*\[(.*?)\];', text, re.S).group(1)
        rows = [row.split() for row in body.split(';') if row.strip()]
        assert len(rows) == 91 and all(len(row) == 5 for row in rows), name
        tables[name] = [' '.join(row) for row in rows]
    return tables


def gmf_tables(text):
    """Return {name: 11 row strings of 5 values, %e} of the DATA statements of gmf.f."""
    tables = {}
    for name in GMF_NAMES:
        body = re.search(r'data\s*\(\s*' + name + r'\(i\),i=1,55\)\s*/(.*?)/', text, re.S | re.I).group(1)
        values = ['%e' % float(v.replace('d', 'e').replace('D', 'e')) for v in NUMBER.findall(body)]
        assert len(values) == 55, name
        tables[name] = [' '.join(values[i:i + 5]) for i in range(0, 55, 5)]
    return tables


def write(path, header, tables):
    """Write the header lines and the [name] blocks of tables to path."""
    lines = list(header)
    for name, rows in tables.items():
        lines.append('[' + name + ']')
        lines += rows
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    """Write both tables from the two source files given on the command line."""
    outdir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(os.path.abspath(__file__))
    with open(sys.argv[1], encoding='latin-1') as f:
        vmf3 = vmf3_tables(f.read())
    with open(sys.argv[2], encoding='latin-1') as f:
        gmf = gmf_tables(f.read())
    write(os.path.join(outdir, 'vmf3_grid.txt'), VMF3_HEADER, vmf3)
    write(os.path.join(outdir, 'gmf.txt'), GMF_HEADER, gmf)
    for name in ('vmf3_grid.txt', 'gmf.txt'):
        path = os.path.join(outdir, name)
        print(name, os.path.getsize(path), hashlib.sha256(open(path, 'rb').read()).hexdigest())


if __name__ == '__main__':
    main()
