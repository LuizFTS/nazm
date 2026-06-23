"""
Module: capture.py

Purpose:
--------
Provides interactive region selection and screenshot capture functionality
for template generation within the application.

Architectural Role:
-------------------
This module operates as an infrastructure-level utility responsible for:
- User-driven region selection via a transient overlay UI (Tkinter).
- Screen capture using MSS.
- Image processing via OpenCV.
- Persisting image templates to a predefined directory in APPDATA.

It is tightly coupled to:
- Windows environment (APPDATA usage and transparent window handling).
- GUI runtime (Tkinter main loop).
- Native screen capture (mss).

This module has side effects:
- Opens a blocking GUI window.
- Captures screen contents.
- Writes files to disk.
- Reads from environment variables.

Design Notes:
-------------
- Procedural utility module with a UI helper class (RegionSelector).
- No dependency injection; direct instantiation of infrastructure dependencies.
- Not platform-agnostic due to transparent color and APPDATA usage.
"""

import os
import tkinter as tk
import uuid
from pathlib import Path

import cv2
import mss
import numpy as np


class RegionSelector:
    """
    UI Component responsible for interactive region selection.

    Responsibilities:
    -----------------
    - Creates a full-screen transparent overlay.
    - Allows click-and-drag rectangular region selection.
    - Translates local window coordinates into global screen coordinates.
    - Stores the final selection as (x, y, width, height).

    Architectural Pattern:
    ----------------------
    - Acts as a UI utility class.
    - Encapsulates user interaction state and geometry normalization logic.
    - Follows event-driven paradigm via Tkinter bindings.

    Important Invariants:
    ---------------------
    - self.selection is either None or a 4-tuple:
        (global_x, global_y, width, height)
    - Coordinates are normalized so width and height are always positive.
    """

    def __init__(self):
        """
        Initializes the full-screen overlay and binds mouse/keyboard events.

        Side Effects:
        -------------
        - Instantiates a Tk root window.
        - Queries monitor geometry using MSS.
        - Creates a transparent, always-on-top overlay window.
        - Binds input events.

        Platform Assumptions:
        ---------------------
        - Assumes support for '-transparentcolor' and '-alpha' attributes
          (commonly supported on Windows).
        """
        self.root = tk.Tk()

        # Retrieve virtual desktop geometry (all monitors combined)
        # mss.monitors[0] represents the bounding box of all monitors.
        with mss.mss() as sct:
            all_monitors = sct.monitors[0]
            self.width = all_monitors["width"]
            self.height = all_monitors["height"]
            self.left = all_monitors["left"]
            self.top = all_monitors["top"]

        # Configure window to cover the entire virtual desktop
        self.root.geometry(f"{self.width}x{self.height}+{self.left}+{self.top}")

        # Remove title bar and window borders for overlay behavior
        self.root.overrideredirect(True)

        # Ensure overlay remains above other windows
        self.root.attributes("-topmost", True)

        # Transparent color configuration.
        # Any region filled with this color becomes fully transparent.
        # This creates the "cut-out" effect in the overlay.
        TRANS_COLOR = "#abcdef"
        self.root.attributes("-transparentcolor", TRANS_COLOR)

        # Canvas serves as the drawing surface for selection.
        # Black background combined with alpha produces dimmed overlay effect.
        self.canvas = tk.Canvas(
            self.root, cursor="cross", bg="black", highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        # Semi-transparent overlay effect
        self.root.attributes("-alpha", 0.6)

        # Interaction state
        self.start_x = None
        self.start_y = None
        self.selection = None

        # Rectangle representing transparent cut-out area
        self.rect_fill = self.canvas.create_rectangle(
            0, 0, 0, 0, fill=TRANS_COLOR, outline=""
        )
        # Separate rectangle for visible border
        self.rect_border = self.canvas.create_rectangle(
            0, 0, 0, 0, outline="red", width=2
        )

        # Event bindings (event-driven interaction model)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Escape key cancels operation
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    def on_button_press(self, event):
        """
        Handles mouse button press event.

        Captures initial anchor coordinates for selection rectangle.

        Inputs:

        -------

        event.x, event.y (local canvas coordinates)
        """
        self.start_x = event.x
        self.start_y = event.y

    def on_move_press(self, event):
        """
        Handles mouse drag (button held).

        Dynamically updates:
        - Transparent fill rectangle (selection cut-out)
        - Visible red border rectangle

        Coordinates are normalized to support dragging in any direction
        (top-left → bottom-right, bottom-right → top-left, etc.).
        """

        cur_x, cur_y = (event.x, event.y)

        # Normalize coordinates to ensure x1 < x2 and y1 < y2
        x1, x2 = sorted([self.start_x, cur_x])
        y1, y2 = sorted([self.start_y, cur_y])

        # Update transparent cut-out rectangle
        self.canvas.coords(self.rect_fill, x1, y1, x2, y2)

        # Update visible border rectangle
        self.canvas.coords(self.rect_border, x1, y1, x2, y2)

    def on_button_release(self, event):
        """
        Finalizes region selection.

        Computes normalized coordinates and converts them
        from local canvas space into global desktop coordinates.

        Output:
        -------
        self.selection:
            (global_x, global_y, width, height)

        Side Effect:
        ------------
        Destroys the overlay window, ending the Tk main loop.
        """
        end_x, end_y = (event.x, event.y)

        # Normalize coordinates to avoid negative dimensions
        x1, x2 = sorted([self.start_x, end_x])
        y1, y2 = sorted([self.start_y, end_y])

        # Translate local window coordinates into global desktop coordinates
        self.selection = (
            int(x1 + self.left),
            int(y1 + self.top),
            int(x2 - x1),
            int(y2 - y1),
        )
        self.root.destroy()


def interactive_capture():
    """
    Orchestrates interactive region selection and screenshot capture.

    Workflow:
    ---------
    1. Launch RegionSelector overlay.
    2. Block until selection completes.
    3. Capture selected region using MSS.
    4. Convert raw image buffer into OpenCV BGR format.
    5. Persist image into APPDATA/nazm/templates.
    6. Return generated filename.

    Returns:
    --------
    - str: filename of saved PNG
    - None: if selection invalid or too small

    External Side Effects:
    ----------------------
    - Opens GUI window.
    - Captures screen pixels.
    - Writes image file to disk.

    Performance Notes:
    ------------------
    - MSS capture is fast and memory-efficient.
    - Conversion BGRA → BGR avoids alpha channel persistence.
    """
    selector = RegionSelector()
    selector.root.mainloop()

    # Guard clause: ignore very small selections (noise protection)
    if not selector.selection or selector.selection[2] < 5:
        return None

    x, y, w, h = selector.selection

    with mss.mss() as sct:
        # MSS expects global monitor coordinates
        monitor = {"top": y, "left": x, "width": w, "height": h}
        sct_img = sct.grab(monitor)

        # Convert raw MSS image into NumPy array
        img = np.array(sct_img)

        # Convert BGRA (MSS default) → BGR (OpenCV standard)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Resolve template directory inside APPDATA
    # Implicit assumption: APPDATA environment variable exists (Windows)
    save_dir = Path(os.getenv("APPDATA")) / "nazm" / "templates"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate collision-resistant filename using UUID
    random_name = f"cap_{uuid.uuid4().hex[:8]}.png"
    full_path = save_dir / random_name

    # Persist image to disk
    cv2.imwrite(str(full_path), img)

    return random_name


def rename_template(old_name: str, new_name: str):
    """
    Renames an existing template file within the APPDATA/nazm/templates directory.

    Responsibilities:
    -----------------
    - Resolves the templates directory path.
    - Validates and normalizes the new filename (ensures .png extension).
    - Performs filesystem rename operation.
    - Returns the new filename on success or None on failure.

    Inputs:
    -------
    old_name : str
        Existing filename (expected to include extension).
    new_name : str
        Desired new filename (extension optional; enforced as .png).

    Returns:
    --------
    str  -> New filename if rename succeeds.
    None -> If the original file does not exist.

    Side Effects:
    -------------
    - Performs a filesystem mutation (rename operation).
    - May overwrite existing file with the same name (platform-dependent behavior).
    - Writes status messages to stdout.

    Assumptions:
    ------------
    - APPDATA environment variable exists (Windows environment).
    - Caller provides filename only (not full path traversal).
    - No validation is performed against path traversal or invalid characters.
    """

    # Resolve base directory for templates.
    # Implicit assumption: Windows environment with APPDATA defined.
    save_dir = Path(os.getenv("APPDATA")) / "nazm" / "templates"

    # Construct absolute path for the existing file.
    old_path = save_dir / old_name

    # Enforce PNG extension to maintain consistency with template storage format.
    # This ensures naming normalization even if caller omits extension.
    if not new_name.lower().endswith(".png"):
        new_name += ".png"

    # Construct target path for rename operation.
    new_path = save_dir / new_name

    # Guard clause: verify source file exists before attempting rename.
    # Prevents raising FileNotFoundError and provides controlled failure path.
    if old_path.exists():
        # Perform atomic rename operation within same filesystem.
        # On Windows, Path.rename() will overwrite if target exists.
        # No collision prevention or versioning strategy is implemented here.
        old_path.rename(new_path)

        # Operational feedback (stdout side effect).
        print(f"Sucesso: {old_name} renomeado para {new_name}")

        return new_name
    else:
        # Explicit failure branch for missing source file.
        print(f"Erro: O arquivo {old_name} não foi encontrado em {save_dir}")
        return None


def list_templates():
    """
    Lists all PNG template files stored in APPDATA/nazm/templates.

    Returns:
    --------
    List[Path]: file paths for each template.

    Behavior:
    ---------
    - Returns empty list if directory does not exist.
    - Filters only '.png' files.
    """
    save_dir = Path(os.getenv("APPDATA")) / "nazm" / "templates"

    # Defensive check: directory may not exist on first run
    if not save_dir.exists():
        return []

    extensions = ".png"

    return [
        p for p in save_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions
    ]


def load_templates():
    """
    Dynamically maps template file paths into attributes
    of a lightweight container object.

    Example:
    --------
    If file 'button_ok.png' exists, result will expose:
        assets.button_ok = "absolute/path/to/button_ok.png"

    Architectural Observation:
    ---------------------------
    - This uses dynamic attribute assignment instead of
      returning a dictionary.
    - Favors dot-notation access over key-based access.

    Risks / Assumptions:
    --------------------
    - File stems must be valid Python attribute names.
    - Name collisions overwrite silently.
    - No validation or sanitization performed.
    """

    class TemplateAssets:
        """
        Empty container class used as a dynamic namespace.
        """

        pass

    assets = TemplateAssets()

    # Iterate through discovered templates and assign attributes dynamically
    for path_obj in list_templates():
        var_name = path_obj.stem
        abs_path = str(path_obj.absolute())

        img_matrix = cv2.imread(abs_path)

        if img_matrix is None:
            print(f"[WARNING] Falha ao carregar template: {abs_path}")

        # Dynamically inject attribute:
        # assets.<filename_without_extension> = absolute_path
        setattr(assets, var_name, img_matrix)
    return assets
