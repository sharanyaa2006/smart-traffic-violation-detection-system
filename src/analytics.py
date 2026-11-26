"""
Phase 6 — Analytics window.

Opens as a separate Tkinter Toplevel window showing:
  1. A bar chart of violation counts by type (matplotlib embedded in Tkinter)
  2. A table of all logged violations (Treeview), pulled from MySQL
  3. Snapshot thumbnails for each violation (clickable to view full size)

Can be launched standalone (python src/analytics.py) or opened from the
main GUI via a button (see gui.py's "View Analytics" button).
"""

import os
import sys
from tkinter import Toplevel, Frame, Label, Tk, Button, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import database


class AnalyticsWindow(Toplevel):
    def __init__(self, master=None, session_start=None):
        super().__init__(master)
        self.title("Analytics — Violation Summary")
        self.geometry("900x700")

        # If a session_start time is given, default view shows only
        # violations logged from this run onward ("New Violations").
        # Older ones stay in the database untouched and are viewable via
        # the "Show All History" toggle — nothing is deleted, just filtered.
        self.session_start = session_start
        self.showing_all_history = False

        self._build_chart_section()
        self._build_table_section()
        self.refresh()

    def _build_chart_section(self):
        chart_frame = Frame(self)
        chart_frame.pack(side="top", fill="x", padx=8, pady=8)

        self.title_label = Label(chart_frame, text="Violations by Type", font=("Arial", 12, "bold"))
        self.title_label.pack(anchor="w")

        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="x")

        btn_frame = Frame(chart_frame)
        btn_frame.pack(anchor="e", pady=4)

        self.toggle_btn = Button(btn_frame, text="Show All History", command=self.toggle_view)
        self.toggle_btn.pack(side="left", padx=4)
        if self.session_start is None:
            self.toggle_btn.config(state="disabled")  # no session context to toggle from

        Button(btn_frame, text="Refresh", command=self.refresh).pack(side="left", padx=4)

    def toggle_view(self):
        self.showing_all_history = not self.showing_all_history
        self.toggle_btn.config(
            text="Show Current Session" if self.showing_all_history else "Show All History"
        )
        self.refresh()

    def _build_table_section(self):
        table_frame = Frame(self)
        table_frame.pack(side="top", fill="both", expand=True, padx=8, pady=8)

        Label(table_frame, text="Violation Log with Snapshots", font=("Arial", 12, "bold")).pack(anchor="w")

        # Create a frame for the table and scrollbars
        tree_frame = Frame(table_frame)
        tree_frame.pack(fill="both", expand=True)

        columns = ("id", "type", "timestamp", "track_id", "snapshot")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=12)
        
        # Configure column headings and widths
        self.tree.heading("id", text="ID")
        self.tree.heading("type", text="Type")
        self.tree.heading("timestamp", text="Timestamp")
        self.tree.heading("track_id", text="Track ID")
        self.tree.heading("snapshot", text="Snapshot")
        
        self.tree.column("id", width=40)
        self.tree.column("type", width=100)
        self.tree.column("timestamp", width=140)
        self.tree.column("track_id", width=60)
        self.tree.column("snapshot", width=100)
        
        # Add scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscroll=vsb.set, xscroll=hsb.set)
        
        # Grid layout
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Bind double-click to open full image
        self.tree.bind("<Double-1>", self._on_row_double_click)
        
        # Store photo references
        self.photo_references = {}
        
        Label(table_frame, text="💡 Double-click snapshot to view full size", 
              font=("Arial", 9), fg="gray").pack(anchor="w", pady=(4, 0))

    def refresh(self):
        since = None if self.showing_all_history else self.session_start
        view_label = "All History" if self.showing_all_history or self.session_start is None else "New Violations (this session)"
        self.title_label.config(text=f"Violations by Type — {view_label}")

        # Update bar chart
        counts = database.get_violation_counts_by_type(since=since)
        self.ax.clear()
        if counts:
            types = list(counts.keys())
            values = list(counts.values())
            self.ax.bar(types, values, color=["#e74c3c", "#e67e22", "#3498db", "#2ecc71"])
            self.ax.set_ylabel("Count")
        else:
            self.ax.text(0.5, 0.5, "No violations logged yet", ha="center", va="center")
        self.fig.tight_layout()
        self.canvas.draw()

        # Update table
        for row in self.tree.get_children():
            self.tree.delete(row)

        # Clear old photo references to prevent garbage collection
        self.photo_references.clear()

        violations = database.get_all_violations(limit=500, since=since)
        for v in violations:
            violation_id = v.get("violation_id")
            snapshot_path = v.get("snapshot_path")
            
            # Create thumbnail if snapshot exists
            thumbnail = self._load_thumbnail(snapshot_path, violation_id)
            
            self.tree.insert("", "end", values=(
                violation_id,
                v.get("type"),
                v.get("timestamp"),
                v.get("track_id"),
                "📷 View" if snapshot_path else "-",
            ), tags=(snapshot_path or "",))
            
            # Store photo reference and path
            if thumbnail:
                self.photo_references[str(violation_id)] = (thumbnail, snapshot_path)

    def _load_thumbnail(self, snapshot_path, violation_id, thumb_size=(80, 60)):
        """Load and create a thumbnail image from snapshot."""
        if not snapshot_path or not os.path.exists(snapshot_path):
            return None
        
        try:
            # Load and resize image
            img = Image.open(snapshot_path)
            img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage for Tkinter
            photo = ImageTk.PhotoImage(img)
            return photo
        except Exception as e:
            print(f"Error loading thumbnail {snapshot_path}: {e}")
            return None

    def _on_row_double_click(self, event):
        """Handle double-click on a row to view full-size snapshot."""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = selection[0]
        values = self.tree.item(item, "values")
        violation_id = values[0]
        
        # Retrieve the snapshot path from photo_references
        if str(violation_id) in self.photo_references:
            _, snapshot_path = self.photo_references[str(violation_id)]
            self._open_full_image_viewer(snapshot_path, violation_id)

    def _open_full_image_viewer(self, snapshot_path, violation_id):
        """Open a new window to display the full-size snapshot."""
        if not os.path.exists(snapshot_path):
            messagebox.showerror("Error", f"Snapshot file not found:\n{snapshot_path}")
            return
        
        try:
            # Create a new window
            viewer_window = Toplevel(self)
            viewer_window.title(f"Violation #{violation_id} Snapshot")
            
            # Load and display the full image
            img = Image.open(snapshot_path)
            
            # Scale to fit screen (max 800x600)
            max_w, max_h = 800, 600
            img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            
            label = Label(viewer_window, image=photo)
            label.image = photo  # Keep a reference
            label.pack(padx=10, pady=10)
            
            # Add file info
            file_info = f"File: {os.path.basename(snapshot_path)}\nSize: {os.path.getsize(snapshot_path) / 1024:.1f} KB"
            info_label = Label(viewer_window, text=file_info, font=("Arial", 9), fg="gray")
            info_label.pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image:\n{str(e)}")


if __name__ == "__main__":
    # Standalone launch for testing: python src/analytics.py
    root = Tk()
    root.withdraw()  # hide the empty root window, only show the analytics Toplevel
    win = AnalyticsWindow(root)
    win.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()