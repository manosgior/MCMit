import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

WIDE_FIGSIZE = 6
COLUMN_FIGSIZE = 3.4
HEIGHT_FIGSIZE = 2.2
FONTSIZE = 12

line_markers = [
    "o",
    "v",
    "s",
    "^",
    "<",
    ">",
    "8",
]

line_styles = [
    "solid",
    "dotted",
    "dashed",
    "dashdot"

]

code_hatches = ["/", "\\", "//", "++", "xx", "**"]

hatches = hatches = [
    "/",
    "\\",
    "//",
    "\\\\",
    "x",
    ".",
    ",",
    "*",
    "o",
    "O",
    "+",
    "X",
    "s",
    "S",
    "d",
    "D",
    "^",
    "v",
    "<",
    ">",
    "p",    
    "P",
    "$",
    "#",
    "%",
]

colors_deep = sns.color_palette("deep")
colors_pastel = sns.color_palette("pastel")