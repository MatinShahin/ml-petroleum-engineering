import nbformat as nbf
import re

def build(cells, title):
    """cells: list of ("md", text) | ("code", code) | ("ex", student_code, solution_code)."""
    stu, sol = nbf.v4.new_notebook(), nbf.v4.new_notebook()
    for c in cells:
        if c[0] == "md":
            stu.cells.append(nbf.v4.new_markdown_cell(c[1].strip())); sol.cells.append(nbf.v4.new_markdown_cell(c[1].strip()))
        elif c[0] == "code":
            stu.cells.append(nbf.v4.new_code_cell(c[1].strip())); sol.cells.append(nbf.v4.new_code_cell(c[1].strip()))
        elif c[0] == "ex":
            stu.cells.append(nbf.v4.new_code_cell(c[1].strip())); sol.cells.append(nbf.v4.new_code_cell(c[2].strip()))
    for nb in (stu, sol):
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["title"] = title
    return stu, sol

SETUP = '''
# --- setup: works locally and in Google Colab ---
import os, sys
if "google.colab" in sys.modules and not os.path.exists("aiml"):
    print("Upload the course kit zip to Colab, then unzip it:  !unzip -q aiml_course_kit.zip && %cd aiml_course_kit")
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
DATA = "../data" if os.path.exists("../data") else "data"
'''
