"""
Smart Traffic Violation Detection System — main entry point.

Run this file to launch the application:
    python main.py

This just launches the Tkinter GUI (src/gui.py). Kept as a separate
top-level entry point so the project has one obvious place to start,
rather than requiring you to know it's src/gui.py specifically.
"""

from tkinter import Tk
from src.gui import SmartTrafficApp

def main():
    root = Tk()
    app = SmartTrafficApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()