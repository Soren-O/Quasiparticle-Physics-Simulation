import tkinter as tk


RETRO_BG = "#C0C0C0"
RETRO_PANEL = "#D4D0C8"
RETRO_BUTTON = "#ECE9D8"
RETRO_ACCENT = "#003399"
RETRO_TEXT = "#111111"

FONT_UI = ("Tahoma", 10)
FONT_TITLE = ("Tahoma", 14, "bold")
FONT_MONO = ("Consolas", 9)


def apply_retro_theme(root: tk.Misc) -> None:
    root.configure(bg=RETRO_BG)
    root.option_add("*Font", FONT_UI)
    root.option_add("*Background", RETRO_BG)
    root.option_add("*Foreground", RETRO_TEXT)
    root.option_add("*Button.Background", RETRO_BUTTON)
    root.option_add("*Button.Relief", "raised")
    root.option_add("*Button.BorderWidth", 2)
    root.option_add("*Entry.Background", "white")
    root.option_add("*Listbox.Background", "white")

