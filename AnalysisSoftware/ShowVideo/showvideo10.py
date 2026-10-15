import cv2
import numpy as np
import time
import os
import sys
import subprocess

# tkinter provides the native "choose a file" and "are you sure?" dialogs
# used by Load mode. It ships with most standard Python installs, but isn't
# guaranteed to be present (e.g. some minimal Linux Python installs need a
# separate `python3-tk` system package) -- if it's missing, Load mode still
# works, just falls back to prompting in the terminal instead of a GUI
# dialog.
try:
  import tkinter as tk
  from tkinter import filedialog as tk_filedialog
  from tkinter import messagebox as tk_messagebox
  TKINTER_AVAILABLE = True
except ImportError:
  TKINTER_AVAILABLE = False

print(
    f"tkinter available: {TKINTER_AVAILABLE}"
    + ("" if TKINTER_AVAILABLE else
       " -- Load mode's file picker will fall back to terminal prompts.")
)

# =============================================================================
# VIDEO PLAYER / ANALYSIS TOOL
# =============================================================================
# Loads an entire video file into memory as a list of frames, then shows it
# in an OpenCV window with a Frame/Brightness/Contrast/Trail-Skip/Stabilize
# trackbar row and a button bar with six modes plus a Help overlay:
#
#   LOAD    - click to choose a different video file to open (with a
#             memory-usage warning if it looks too big to load comfortably).
#   VIEW    - scrub frames, hover to inspect a pixel's raw RGB value.
#   SAVE    - click the video to save a 300x300 crop centered on the click.
#   PLAY    - click to start/stop normal playback at the source frame rate.
#   SEGMENT - click inside a bright object to find its boundary (with
#             sub-pixel precision) and report its size/intensity/position.
#   TRAIL   - combine every frame into one "light trail" composite image,
#             optionally camera-stabilized, and save it to disk.
#
# Layout: the OpenCV window is one tall image (`canvas`) made of three
# vertically-stacked pieces so nothing is ever drawn on top of the video
# picture itself:
#
#   +----------------------------------------------------+
#   |  mode bar (buttons: Load/View/Save/Play/Segment/    |  <- mode_bar_h tall
#   |            Trail/Help)                              |
#   +----------------------------------------------------+
#   |                                                      |
#   |               the video frame itself                |  <- frame_h tall
#   |         (only the Play button is drawn on it)        |
#   |                                                      |
#   +----------------------------------------------------+
#   |  info bar (pixel readout / mode-specific messages)   |  <- info_bar_h tall
#   +----------------------------------------------------+
#
# Mouse coordinates from OpenCV are in on-screen window pixels, which have to
# be rescaled to canvas pixels (the window can be resized) and then shifted
# up by `frame_y_offset` to become "frame-local" coordinates before they can
# be compared against the video frame's own pixel positions. See
# mouse_callback() below for exactly where that happens.
#
# frames_cache/total_frames/frame_h/frame_w/frame_interval/video_path are
# all set by load_video_file() below, and can change at runtime whenever
# Load mode successfully opens a new file -- nothing else in this file
# should assume they're fixed for the whole run.
# =============================================================================


def format_bytes(num_bytes):
  """Human-readable byte count, e.g. 1536000000 -> '1.43 GB'."""
  size = float(num_bytes)
  for unit in ("B", "KB", "MB", "GB", "TB"):
    if size < 1024.0 or unit == "TB":
      return f"{size:.2f} {unit}"
    size /= 1024.0


def get_available_memory_bytes():
  """Best-effort estimate of total system RAM in bytes, for the Load
  memory-safety check. Uses os.sysconf, which works on Linux/macOS; falls
  back to a conservative fixed assumption on platforms (e.g. Windows) or
  environments where that's unavailable, since this script has no required
  dependency (like psutil) that could query it more precisely.
  """
  try:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
  except (ValueError, AttributeError, OSError):
    ASSUMED_RAM_BYTES = 8 * (1024 ** 3)  # assume a conservative 8 GB
    return ASSUMED_RAM_BYTES


def probe_video_metadata(path):
  """Read cheap metadata (frame count, width, height, fps) from a video
  file WITHOUT decoding any actual frame pixels. Returns a dict, or None
  if the file can't even be opened.
  """
  probe_cap = cv2.VideoCapture(path)
  if not probe_cap.isOpened():
    probe_cap.release()
    return None
  metadata = {
      "frame_count": int(probe_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
      "width": int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
      "height": int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
      "fps": probe_cap.get(cv2.CAP_PROP_FPS),
  }
  probe_cap.release()
  return metadata


def estimate_memory_check(path):
  """Decide whether loading `path` looks safe to fit comfortably in memory.

  Compressed on-disk file size is a poor proxy for actual memory usage --
  this app stores every frame fully decoded (uncompressed) in RAM, and a
  modestly-sized h.264/HEVC file can easily decode to many times its
  compressed size. So the estimate is based on video METADATA (frame
  count x width x height x 3 bytes/pixel for BGR), read without decoding
  anything, and only falls back to the raw file size if that metadata
  isn't available (some containers under-report frame count).

  Returns (is_too_big, message): is_too_big is True if the estimated
  footprint exceeds a comfortable fraction of total system RAM; message is
  a human-readable explanation to show the user either way.
  """
  try:
    file_size_bytes = os.path.getsize(path)
  except OSError:
    file_size_bytes = None

  metadata = probe_video_metadata(path)
  available_bytes = get_available_memory_bytes()
  # Never plan to use more than this fraction of total system RAM for the
  # frame cache -- leaves headroom for the OS, this Python process's own
  # overhead, and everything else running on the machine.
  COMFORTABLE_FRACTION = 0.4
  comfortable_limit_bytes = available_bytes * COMFORTABLE_FRACTION

  have_reliable_metadata = (
      metadata is not None
      and metadata["frame_count"] > 0
      and metadata["width"] > 0
      and metadata["height"] > 0
  )
  if have_reliable_metadata:
    BYTES_PER_PIXEL_BGR = 3
    estimated_bytes = (
        metadata["frame_count"] * metadata["width"] * metadata["height"] * BYTES_PER_PIXEL_BGR
    )
    basis = "estimated from decoded frame count x width x height"
  else:
    # Frame count metadata is missing/unreliable for this container --
    # fall back to the compressed file size itself. This is usually a
    # considerable UNDER-estimate (decoded frames are almost always much
    # bigger than their compressed form), so treat it as a rough floor.
    estimated_bytes = file_size_bytes or 0
    basis = "estimated from compressed file size (frame-count metadata unavailable -- likely an UNDER-estimate)"

  is_too_big = estimated_bytes > comfortable_limit_bytes

  message = (
      f"File size on disk: {format_bytes(file_size_bytes) if file_size_bytes else 'unknown'}\n"
      f"Estimated memory needed once loaded: ~{format_bytes(estimated_bytes)}\n"
      f"  ({basis})\n"
      f"Comfortable limit for this machine: ~{format_bytes(comfortable_limit_bytes)}"
  )
  return is_too_big, message


def confirm_dialog(title, message):
  """Ask a yes/no question via a native dialog (tkinter) if available,
  otherwise fall back to a terminal prompt. Always returns a bool.
  """
  if TKINTER_AVAILABLE:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
      return tk_messagebox.askyesno(title, message)
    finally:
      root.destroy()
  print(f"\n{title}\n{message}")
  answer = input("Proceed? [y/N]: ").strip().lower()
  return answer.startswith("y")


def report_load_problem(message):
  """Surface a load failure to the user: always to the console, and also
  as a native error dialog when tkinter is available.
  """
  print(f"Load error: {message}")
  if TKINTER_AVAILABLE:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
      tk_messagebox.showerror("Load error", message)
    finally:
      root.destroy()


def try_macos_native_file_picker():
  """On macOS, ask the OS's own native file-picker dialog (via osascript/
  AppleScript) to choose a file -- no Python GUI package (tkinter, Qt,
  etc.) required at all, since AppleScript is part of every macOS install.

  Returns:
    - a non-empty path string: the user picked a file.
    - "" (empty string): the dialog was shown and the user cancelled it.
    - None: not on macOS, or osascript itself couldn't be run/failed to
      start (e.g. no GUI session) -- caller should fall back to another
      method entirely.
  """
  if sys.platform != "darwin":
    return None

  # This AppleScript shows the standard native "choose a file" dialog and
  # converts the result to a normal POSIX path string (the dialog's raw
  # result is a classic Mac "alias", e.g. "Macintosh HD:Users:name:x.mov:").
  applescript = 'POSIX path of (choose file with prompt "Choose a video file")'
  try:
    result = subprocess.run(
        ["osascript", "-e", applescript],
        capture_output=True, text=True, timeout=300,
    )
  except (OSError, subprocess.TimeoutExpired):
    # osascript missing/unrunnable -- fall back to another method rather
    # than fail outright.
    return None

  if result.returncode != 0:
    # AppleScript raises an error when the user clicks Cancel (or the
    # dialog fails for some other reason); either way, no path was chosen.
    return ""

  return result.stdout.strip()


def prompt_for_video_path():
  """Get a video file path from the user, trying progressively simpler
  fallbacks so Load mode works regardless of what's installed:
    1. On macOS: the OS's own native file dialog (no dependency at all).
    2. tkinter's file dialog, if available.
    3. A plain terminal prompt to type/paste a path directly.

  Returns "" (empty string) if the user cancelled/left it blank, or a
  non-empty path string otherwise.
  """
  native_result = try_macos_native_file_picker()
  if native_result is not None:
    print(f"Native file picker closed. Chosen path: {native_result!r}")
    return native_result

  if not TKINTER_AVAILABLE:
    print(
        "No native file picker available and tkinter isn't installed, so "
        "Load mode can't show a GUI file browser. Falling back to a "
        "terminal prompt instead -- type or paste a path below. (On "
        "Linux, installing tkinter -- often a separate system package, "
        "e.g. 'python3-tk' -- would give you a proper file-browser "
        "dialog instead.)"
    )
    try:
      typed_path = input(
          "Enter the path to a video file (leave blank to cancel): "
      )
    except EOFError:
      # No interactive terminal attached (e.g. launched without a console)
      # -- there's nowhere to type into, so treat it as a cancel.
      print("No interactive terminal available to type a path into.")
      return ""
    # Be forgiving of how the path arrives: strip surrounding whitespace
    # and quote characters (a common artifact of dragging a file into
    # some terminals, which wraps the path in quotes), and expand "~".
    typed_path = typed_path.strip().strip('"').strip("'")
    typed_path = os.path.expanduser(typed_path)
    return typed_path

  print("Opening file picker for Load... (check for a new window/dialog "
        "that may have opened behind this one)")
  root = tk.Tk()
  root.withdraw()
  # Best-effort attempt to make sure the dialog actually comes to the
  # front instead of opening silently behind the OpenCV window -- window
  # manager behavior here varies a fair bit across OSes/desktop
  # environments, so this isn't guaranteed, but it doesn't hurt.
  root.attributes("-topmost", True)
  root.update()
  root.lift()
  root.focus_force()
  try:
    chosen_path = tk_filedialog.askopenfilename(
        parent=root,
        title="Choose a video file",
        filetypes=[
            (
                "Video files",
                "*.mov *.mp4 *.m4v *.avi *.mkv *.wmv *.flv *.webm "
                "*.mpg *.mpeg *.3gp *.3g2 *.ts *.m2ts",
            ),
            ("All files", "*.*"),
        ],
    )
  finally:
    root.destroy()

  print(f"File picker closed. Chosen path: {chosen_path!r}")
  return chosen_path  # "" if the user cancelled; a path string otherwise


def print_frame_timing_diagnostics(loaded_frames, timestamps_ms):
  """Diagnose actual per-frame timing from the timestamps OpenCV read off
  each frame, since iPhone .mov files (especially Live Photo sidecars) are
  often Variable Frame Rate and the single "reported FPS" from the
  container's metadata can disagree with what any given app displays.
  """
  if len(timestamps_ms) <= 1:
    return
  intervals_ms = [
      timestamps_ms[i] - timestamps_ms[i - 1]
      for i in range(1, len(timestamps_ms))
  ]
  min_interval = min(intervals_ms)
  max_interval = max(intervals_ms)
  avg_interval = sum(intervals_ms) / len(intervals_ms)
  duration_s = timestamps_ms[-1] / 1000.0
  frame_count = len(loaded_frames)
  avg_fps_from_timestamps = (
      (frame_count - 1) / duration_s if duration_s > 0 else 0.0
  )

  print(f"Frame count: {frame_count}")
  print(f"Duration (from timestamps): {duration_s:.3f} s")
  print(
      f"Per-frame interval (ms) — min: {min_interval:.2f}, "
      f"max: {max_interval:.2f}, avg: {avg_interval:.2f}"
  )
  print(f"Average FPS derived from timestamps: {avg_fps_from_timestamps:.2f}")

  if max_interval - min_interval > 1.0:  # more than ~1ms spread
    print(
        "-> Frame intervals are not constant: this looks like a Variable "
        "Frame Rate (VFR) file, so no single FPS value fully describes it."
    )
  else:
    print("-> Frame intervals are essentially constant (constant frame rate).")


def load_video_file(path, interactive=True):
  """Load `path` as the active video: decode every frame into memory and
  replace the module-level frames_cache/total_frames/frame_h/frame_w/
  frame_interval/video_path with the newly loaded video's data.

  If `interactive` is True and the estimated memory footprint looks too
  big to fit comfortably, asks the user to confirm before proceeding
  (native dialog if tkinter is available, else a terminal prompt);
  declining aborts the load and leaves whatever was previously loaded
  completely untouched. Returns True on a successful load, False
  otherwise (file missing/unreadable/empty, or the user declined).
  """
  global frames_cache, total_frames, frame_h, frame_w, frame_interval, video_path

  if not os.path.isfile(path):
    report_load_problem(f"File not found:\n{path}")
    return False

  is_too_big, memory_message = estimate_memory_check(path)
  if is_too_big:
    proceed = confirm_dialog(
        "Large video file",
        f"{memory_message}\n\n"
        f"This may use a large amount of memory and could make your "
        f"system unresponsive while it loads. Load it anyway?",
    ) if interactive else True
    if not proceed:
      print(f"Load cancelled by user (file too large): {path}")
      return False

  cap = cv2.VideoCapture(path)
  if not cap.isOpened():
    cap.release()
    report_load_problem(f"Could not open video file:\n{path}")
    return False

  fps = cap.get(cv2.CAP_PROP_FPS)
  if fps <= 0:
    fps = 30.0

  print(f"Loading frames from {path} ... Reported Video FPS: {fps:.2f}")
  loaded_frames = []
  loaded_timestamps_ms = []
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    loaded_frames.append(frame)
    loaded_timestamps_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
  cap.release()

  if len(loaded_frames) == 0:
    report_load_problem(f"No frames could be read from:\n{path}")
    return False

  print_frame_timing_diagnostics(loaded_frames, loaded_timestamps_ms)

  frames_cache = loaded_frames
  total_frames = len(loaded_frames)
  frame_h, frame_w = loaded_frames[0].shape[:2]
  frame_interval = 1.0 / fps
  video_path = path
  return True


DEFAULT_VIDEO_PATH = "./2025-10-30-CPOfficeCloudsOrbs2.mov"
DEFAULT_VIDEO_PATH = "./2ECDBF66-F120-414B-9953-B1DD00EC186B.mov"
if not load_video_file(DEFAULT_VIDEO_PATH, interactive=True):
  print("Error: could not load the initial video. Exiting.")
  exit()

# --- Mode bar (a strip ABOVE the video image, kept off the picture itself,
# same idea as the RGB info bar below it) holding the mode buttons: View,
# Image Save, Play, Segment, and Trail, plus the Load and Help actions. ---
MODE_VIEW = "VIEW"
MODE_SAVE = "IMAGE_SAVE"
MODE_PLAY = "PLAY"
MODE_SEGMENT = "SEGMENT"
MODE_TRAIL = "TRAIL"
current_mode = MODE_VIEW

margin = 10
mode_bar_h = 56
mode_btn_w, mode_btn_h = 96, 36
mode_btn_gap = 5

# LOAD (like HELP) isn't one of the mutually-exclusive modes above -- it's
# a one-shot action button. Clicking it immediately opens the file picker
# rather than switching into a lingering "Load mode", since unlike
# Save/Segment/Trail its action doesn't depend on where you click.
load_btn_x1 = margin
load_btn_y1 = (mode_bar_h - mode_btn_h) // 2
load_btn_x2 = load_btn_x1 + mode_btn_w
load_btn_y2 = load_btn_y1 + mode_btn_h

view_btn_x1 = load_btn_x2 + mode_btn_gap
view_btn_y1 = load_btn_y1
view_btn_x2 = view_btn_x1 + mode_btn_w
view_btn_y2 = view_btn_y1 + mode_btn_h

save_btn_x1 = view_btn_x2 + mode_btn_gap
save_btn_y1 = view_btn_y1
save_btn_x2 = save_btn_x1 + mode_btn_w
save_btn_y2 = save_btn_y1 + mode_btn_h

play_btn_x1 = save_btn_x2 + mode_btn_gap
play_btn_y1 = view_btn_y1
play_btn_x2 = play_btn_x1 + mode_btn_w
play_btn_y2 = play_btn_y1 + mode_btn_h

segment_btn_x1 = play_btn_x2 + mode_btn_gap
segment_btn_y1 = view_btn_y1
segment_btn_x2 = segment_btn_x1 + mode_btn_w
segment_btn_y2 = segment_btn_y1 + mode_btn_h

trail_btn_x1 = segment_btn_x2 + mode_btn_gap
trail_btn_y1 = view_btn_y1
trail_btn_x2 = trail_btn_x1 + mode_btn_w
trail_btn_y2 = trail_btn_y1 + mode_btn_h

help_btn_x1 = trail_btn_x2 + mode_btn_gap
help_btn_y1 = view_btn_y1
help_btn_x2 = help_btn_x1 + mode_btn_w
help_btn_y2 = help_btn_y1 + mode_btn_h

# --- Help overlay ---
# Not a "mode" like the others -- it's a dismissable overlay shown on top
# of whatever's currently on screen, listing every mode/control/behavior.
show_help = False

# --- Image Save mode settings ---
CAPTURE_SIZE = 300
CAPTURE_DIR = "./captures"
save_counter = 0
last_save_message = ""
last_save_time = 0.0
SAVE_MESSAGE_DURATION_S = 1.5

# --- Segment mode settings ---
# Result of the most recent segmentation attempt, and the frame index it was
# computed on (so the overlay/report is dropped if the frame moves on and it
# no longer applies).
segment_result = None
segment_frame_index = -1
segment_message = "Segment mode: click inside a bright object"
segment_version = 0  # bumped on every click so the draw loop knows to redraw

# --- Trail mode settings ---
# A "light trail" composite: every cached frame combined with a per-pixel
# maximum (lighten blend), so anything that moved leaves a bright trail
# against a static background. Computed once and cached, since frames_cache
# never changes during the run.
TRAIL_DIR = "./trails"
trail_raw_frame = None  # the raw (unadjusted) max-projection composite
trail_raw_skip = None   # the skip-frames value that composite was computed for
trail_raw_count = None  # the frame-count value that composite was computed for
trail_raw_stabilize = None  # the stabilize flag that composite was computed for
# (skip, count, stabilize, brightness, contrast) tuple of the last
# combination that was actually written to disk, so we don't save an
# identical duplicate.
last_trail_saved_key = None

# A strip below the video image, used to show the original RGB readout
# (kept fully off the picture itself instead of overlaid on top of it).
info_bar_h = 50

# OpenCV renders each trackbar's label and current-value text within the
# window's own client width, cut off if that's too narrow. Our longest
# trackbar label ("Trail Skip") plus its slider and numeric readout
# needs more horizontal room than a narrow (e.g. portrait) video frame
# provides, so the window/canvas width is never allowed to shrink below a
# floor that comfortably fits it -- the video frame itself just stays its
# native width, left-aligned, with any extra width as blank margin.
MIN_WINDOW_W = 960
canvas_w = max(frame_w, MIN_WINDOW_W)
canvas_h = mode_bar_h + frame_h + info_bar_h

# Vertical offset of the video image within the canvas (below the mode bar).
frame_y_offset = mode_bar_h


def on_trackbar(val):
  pass


cv2.namedWindow("Video Player", cv2.WINDOW_NORMAL)

cv2.createTrackbar("Frame", "Video Player", 0, total_frames - 1, on_trackbar)
cv2.createTrackbar("Brightness", "Video Player", 50, 100, on_trackbar)
cv2.createTrackbar("Contrast", "Video Player", 10, 30, on_trackbar)
# How many leading frames to leave out of the Trail composite (frame indices
# 0 .. Trail Skip Frames - 1 are excluded). Max leaves at least one frame in.
cv2.createTrackbar("Trail Skip", "Video Player", 0, max(0, total_frames - 1), on_trackbar)
# How many frames (starting at Trail Skip) to actually combine. Defaults to
# every remaining frame (the previous, only, behavior), but max-blending
# hundreds of frames of ordinary well-lit/moving footage tends to wash the
# whole composite out into a smeared mess rather than a clean trail -- this
# lets it be narrowed down to just the handful of frames that actually
# matter.
cv2.createTrackbar("Trail Count", "Video Player", total_frames, total_frames, on_trackbar)
# 0 = off, 1 = on. Aligns each frame back to a common reference before
# combining, so camera motion doesn't smear/ghost the trail composite.
cv2.createTrackbar("Stabilize", "Video Player", 0, 1, on_trackbar)

current_frame = -1
is_playing = False
last_frame_time = time.time()

prev_brightness = -1
prev_contrast = -1
prev_mode = current_mode
prev_is_playing = is_playing
prev_segment_version = -1
prev_trail_skip = -1
prev_trail_count = -1
prev_stabilize = -1
prev_show_help = False

# Mouse tracking for the original-pixel RGB readout / crop preview.
# These are stored in FRAME-LOCAL coordinates (i.e. already adjusted for
# the mode bar sitting above the video image in the canvas).
mouse_x, mouse_y = -1, -1
mouse_in_frame = False

# Updated every redraw so the mouse callback can save a clean (no button/
# text overlay) crop of whatever is currently on screen, and so Segment
# mode can run on the same pixels the user is looking at.
latest_original_frame = frames_cache[0]
latest_adjusted_frame = frames_cache[0]


def save_crop(source_frame, center_x, center_y, size):
  """Save a size x size crop of source_frame centered on (center_x, center_y).

  The click point is always exactly centered in the output, even right at
  the edge/corner of the frame. Whatever part of the size x size window
  falls outside the source frame is padded with black rather than shifting
  the window to stay in-bounds.
  """
  source_h, source_w = source_frame.shape[:2]
  half_size = size // 2

  # The crop window we WANT, expressed in source-frame coordinates. This
  # can extend past the source frame's edges (negative, or beyond
  # source_w/source_h) -- that's fine, it just means part of the output
  # will end up black.
  wanted_x1 = center_x - half_size
  wanted_y1 = center_y - half_size
  wanted_x2 = wanted_x1 + size
  wanted_y2 = wanted_y1 + size

  # Clip that wanted window down to the portion that actually exists
  # within the source frame's real pixel bounds.
  overlap_x1 = max(0, wanted_x1)
  overlap_y1 = max(0, wanted_y1)
  overlap_x2 = min(source_w, wanted_x2)
  overlap_y2 = min(source_h, wanted_y2)

  output_image = np.zeros((size, size, 3), dtype="uint8")

  if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
    # Work out where that overlapping region should land inside the
    # (always size x size) output image -- i.e. how far the wanted
    # window's top-left corner got clipped, which is exactly how far the
    # valid pixels need to be shifted inward from the output's own corner.
    paste_x1 = overlap_x1 - wanted_x1
    paste_y1 = overlap_y1 - wanted_y1
    paste_x2 = paste_x1 + (overlap_x2 - overlap_x1)
    paste_y2 = paste_y1 + (overlap_y2 - overlap_y1)
    output_image[paste_y1:paste_y2, paste_x1:paste_x2] = (
        source_frame[overlap_y1:overlap_y2, overlap_x1:overlap_x2]
    )

  os.makedirs(CAPTURE_DIR, exist_ok=True)
  filename = f"capture_{save_counter:03d}.png"
  filepath = os.path.join(CAPTURE_DIR, filename)
  cv2.imwrite(filepath, output_image)
  return filename


def segment_bright_object(gray_raw, seed_x, seed_y):
  """Find the boundary of a bright object at (seed_x, seed_y) in a raw
  grayscale image, assuming it's brighter than its surrounding background.

  Two-stage approach:
    1. Coarse localization: estimate the object's local intensity from a
       small patch around the seed, estimate the background as a low
       percentile of the whole frame, threshold halfway between the two,
       and take the connected "bright" component touching the seed. This
       gives an approximate bounding box.
    2. Sub-pixel refinement: crop a padded region of interest (ROI) around
       that bounding box, run Canny edge detection on it, take the largest
       edge contour, and refine its points to sub-pixel precision with
       cv2.cornerSubPix (which snaps each point to the true
       intensity-gradient edge location within its neighborhood, rather
       than the nearest whole pixel).

  Returns None if no genuinely bright, large-enough object is found at the
  seed, or a dict with the refined boundary, length/width, average
  intensity, and intensity-weighted center of mass otherwise.
  """
  img_h, img_w = gray_raw.shape[:2]

  # ------------------------------------------------------------------
  # Stage 1: coarse localization via simple intensity thresholding.
  # ------------------------------------------------------------------

  # Sample a small square patch centered on the click to estimate how
  # bright the object itself is at that point (more robust to a single
  # noisy pixel than reading just gray_raw[seed_y, seed_x] directly).
  patch_radius = 5
  patch_y1, patch_y2 = max(0, seed_y - patch_radius), min(img_h, seed_y + patch_radius + 1)
  patch_x1, patch_x2 = max(0, seed_x - patch_radius), min(img_w, seed_x + patch_radius + 1)
  object_patch = gray_raw[patch_y1:patch_y2, patch_x1:patch_x2]
  if object_patch.size == 0:
    return None
  object_intensity = float(np.mean(object_patch))

  # Background estimate: a low percentile of the whole frame. Works well
  # when the background occupies most of the frame, which is the common
  # case for an isolated bright object on a darker background.
  background_intensity = float(np.percentile(gray_raw, 25))

  # Minimum contrast gate, expressed as a percentage of the background
  # intensity (Weber contrast: (object - background) / background). A fixed
  # absolute intensity gap doesn't scale with scene brightness -- e.g. a
  # dark background would demand an unreasonably large relative jump -- so
  # this is relative instead, and set low enough to catch fairly subtle
  # objects.
  MIN_CONTRAST_PERCENT = 6.0
  contrast_percent = (
      (object_intensity - background_intensity) / max(background_intensity, 1.0)
  ) * 100.0

  if contrast_percent < MIN_CONTRAST_PERCENT:
    return None  # Not meaningfully brighter than the background at the seed.

  # Binarize the whole frame at the midpoint between the object's and the
  # background's estimated intensity, then find the connected "bright"
  # region that the click actually landed on.
  threshold = (object_intensity + background_intensity) / 2.0
  binary_mask = (gray_raw > threshold).astype("uint8") * 255

  num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
  seed_label = labels[seed_y, seed_x]
  if seed_label == 0:
    return None  # Seed pixel didn't land on a bright connected region.

  # stats[] columns are (x, y, width, height, area) for each labeled blob;
  # this is the axis-aligned bounding box of the coarse detection.
  bbox_x, bbox_y, bbox_w, bbox_h, area = stats[seed_label]
  if area < 25:
    return None  # Too small to be a meaningful object (likely noise).

  # Guard against runaway segmentation: on very low-contrast/noisy frames,
  # the global threshold can sit close enough to the noise floor that
  # scattered background pixels bridge together into one giant connected
  # blob engulfing most of the frame, rather than isolating the actual
  # object. Reject that outright instead of reporting a bogus huge boundary.
  MAX_OBJECT_AREA_FRACTION = 0.25   # object area vs. whole-frame area
  MAX_OBJECT_DIM_FRACTION = 0.9     # object bounding box vs. frame width/height
  frame_area = float(img_h * img_w)
  if area > MAX_OBJECT_AREA_FRACTION * frame_area:
    return None  # Region covers too much of the frame to be a single object.
  if bbox_w > MAX_OBJECT_DIM_FRACTION * img_w or bbox_h > MAX_OBJECT_DIM_FRACTION * img_h:
    return None  # Bounding box spans nearly the whole frame in some axis.

  # A black-and-white mask, same size as the full frame, containing just
  # the connected blob the click landed on (this is the "coarse" result,
  # used as the fallback boundary if sub-pixel refinement below fails).
  coarse_mask = (labels == seed_label).astype("uint8") * 255

  # ------------------------------------------------------------------
  # Stage 2: sub-pixel refinement via Canny + cornerSubPix, confined to
  # a padded region of interest (ROI) around the coarse bounding box
  # (cheaper and more robust than running Canny over the whole frame).
  # ------------------------------------------------------------------
  roi_pad = 12
  roi_x1 = max(0, bbox_x - roi_pad)
  roi_y1 = max(0, bbox_y - roi_pad)
  roi_x2 = min(img_w, bbox_x + bbox_w + roi_pad)
  roi_y2 = min(img_h, bbox_y + bbox_h + roi_pad)
  roi_image = gray_raw[roi_y1:roi_y2, roi_x1:roi_x2]

  refined_ok = False
  if roi_image.shape[0] >= 5 and roi_image.shape[1] >= 5:
    blurred_roi = cv2.GaussianBlur(roi_image, (5, 5), 0)

    # Automatic Canny thresholds derived from the ROI's actual gradient
    # magnitude distribution (Sobel), rather than from raw pixel intensity.
    # An intensity-based heuristic assumes a certain amount of contrast is
    # present; for a genuinely low-contrast object the true edge gradient
    # can be weak even when the background/foreground intensities are far
    # apart in absolute terms elsewhere in the frame, so thresholds must be
    # set relative to the gradients that are actually present in this ROI.
    gradient_x = cv2.Sobel(blurred_roi, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred_roi, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
    canny_upper = max(10.0, float(np.percentile(gradient_magnitude, 90)))
    canny_lower = canny_upper * 0.4
    edge_map = cv2.Canny(blurred_roi, canny_lower, canny_upper)

    edge_contours, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if edge_contours:
      best_contour = max(edge_contours, key=cv2.contourArea)
      if cv2.contourArea(best_contour) >= 10 and len(best_contour) >= 5:
        contour_points = best_contour.reshape(-1, 1, 2).astype(np.float32)

        # Cap the point count for speed; cornerSubPix cost scales with it.
        MAX_REFINE_POINTS = 300
        if len(contour_points) > MAX_REFINE_POINTS:
          sample_idx = np.linspace(0, len(contour_points) - 1, MAX_REFINE_POINTS).astype(int)
          contour_points = contour_points[sample_idx]

        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        try:
          # cornerSubPix nudges each point to the true sub-pixel edge
          # location, based on the local intensity gradient around it.
          refined_points = cv2.cornerSubPix(blurred_roi, contour_points, (5, 5), (-1, -1), subpix_criteria)
          # Map refined points from ROI-local back to full-frame coordinates.
          refined_points[:, 0, 0] += roi_x1
          refined_points[:, 0, 1] += roi_y1
          refined_ok = True
        except cv2.error:
          refined_ok = False

  if refined_ok:
    # Validate the refined polygon before committing to it. cornerSubPix
    # can occasionally collapse points onto a degenerate (near-zero-area or
    # self-intersecting) shape -- or, especially on weak/noisy gradients,
    # wander onto a plausible-looking but WRONG shape that isn't actually
    # degenerate. Catch both: reject the refined result if its area or
    # location disagrees substantially with the coarse detection, so the
    # fallback below is used for drawing (not just for the statistics).
    refined_mask = np.zeros((img_h, img_w), dtype="uint8")
    cv2.fillPoly(refined_mask, [np.round(refined_points).astype(np.int32)], 255)
    refined_area = float(refined_mask.sum()) / 255.0

    if refined_area == 0:
      refined_ok = False
    else:
      area_ratio = refined_area / float(area)
      if area_ratio < 0.3 or area_ratio > 3.0:
        refined_ok = False  # implausible size compared to the coarse blob
      else:
        refined_moments = cv2.moments(refined_mask, binaryImage=True)
        refined_center_x = refined_moments["m10"] / refined_moments["m00"]
        refined_center_y = refined_moments["m01"] / refined_moments["m00"]
        coarse_center_x, coarse_center_y = bbox_x + bbox_w / 2.0, bbox_y + bbox_h / 2.0
        center_drift = (
            (refined_center_x - coarse_center_x) ** 2 + (refined_center_y - coarse_center_y) ** 2
        ) ** 0.5
        allowed_drift = max(bbox_w, bbox_h) * 0.5 + 5
        if center_drift > allowed_drift:
          refined_ok = False  # implausibly far from where the object actually is

  if refined_ok:
    # Use the sub-pixel refined boundary: a float32 polygon in full-frame
    # coordinates.
    boundary_points_float = refined_points
    min_area_rect = cv2.minAreaRect(boundary_points_float)
    object_mask = refined_mask
  else:
    # Fall back to the coarse (whole-pixel) boundary if edge refinement
    # didn't produce a usable contour.
    coarse_contours, _ = cv2.findContours(coarse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not coarse_contours:
      return None
    boundary_points_float = max(coarse_contours, key=cv2.contourArea).astype(np.float32)
    min_area_rect = cv2.minAreaRect(boundary_points_float)
    object_mask = coarse_mask

  # min_area_rect is ((center_x, center_y), (side_a, side_b), angle_degrees)
  # -- the two sides aren't guaranteed to come out as (width, height) in any
  # particular order, so length is just whichever side is longer.
  (rect_side_a, rect_side_b) = min_area_rect[1]
  length = max(rect_side_a, rect_side_b)
  width = min(rect_side_a, rect_side_b)

  # Final belt-and-suspenders check on the actual reported dimensions, in
  # case refinement ever expanded the boundary beyond what the coarse-stage
  # guard above already screened for.
  if length > MAX_OBJECT_DIM_FRACTION * max(img_w, img_h):
    return None

  box_corner_points = cv2.boxPoints(min_area_rect).astype(int)  # kept for reference, not drawn
  boundary_points_int = np.round(boundary_points_float).astype(np.int32).reshape(-1, 1, 2)

  if refined_ok:
    # Weight each pixel by how much of it actually falls inside the
    # sub-pixel boundary, instead of a binary in/out pixel mask, by
    # rasterizing the polygon at a higher resolution and downsampling with
    # area averaging (each output pixel becomes its true fractional
    # coverage, 0.0-1.0).
    roi_gray = gray_raw[roi_y1:roi_y2, roi_x1:roi_x2].astype(np.float32)
    roi_h, roi_w = roi_gray.shape
    supersample_factor = 8

    # Shift the polygon into ROI-local coordinates and scale it up so it
    # can be rasterized at higher-than-pixel resolution.
    polygon_roi_local = boundary_points_float.copy()
    polygon_roi_local[:, 0, 0] -= roi_x1
    polygon_roi_local[:, 0, 1] -= roi_y1
    polygon_supersampled = np.round(polygon_roi_local * supersample_factor).astype(np.int32)

    supersampled_mask = np.zeros((roi_h * supersample_factor, roi_w * supersample_factor), dtype="uint8")
    cv2.fillPoly(supersampled_mask, [polygon_supersampled], 255)
    # Downsampling with area-averaging turns the crisp high-res mask into a
    # per-pixel fractional coverage map (0.0 = fully outside the boundary,
    # 1.0 = fully inside, values in between for edge pixels).
    pixel_coverage = cv2.resize(
        supersampled_mask, (roi_w, roi_h), interpolation=cv2.INTER_AREA
    ).astype(np.float32) / 255.0

    total_coverage_weight = float(pixel_coverage.sum())
    if total_coverage_weight > 0:
      avg_intensity = float((roi_gray * pixel_coverage).sum() / total_coverage_weight)
    else:
      avg_intensity = float(gray_raw[object_mask > 0].mean())

    # Intensity-weighted center of mass (sub-pixel), in full-frame
    # coordinates, using the same fractional pixel-coverage weighting.
    pixel_mass = roi_gray * pixel_coverage
    total_mass = float(pixel_mass.sum())
    if total_mass > 0:
      row_indices, col_indices = np.indices((roi_h, roi_w), dtype=np.float32)
      center_of_mass_x = float((col_indices * pixel_mass).sum() / total_mass) + roi_x1
      center_of_mass_y = float((row_indices * pixel_mass).sum() / total_mass) + roi_y1
    else:
      center_of_mass_x, center_of_mass_y = min_area_rect[0]
  else:
    avg_intensity = float(gray_raw[object_mask > 0].mean())

    # Intensity-weighted center of mass over the coarse binary mask (no
    # fractional pixel coverage available here, so it's a simple weighted
    # average over the whole-pixel mask instead).
    mask_row_indices, mask_col_indices = np.nonzero(object_mask)
    pixel_weights = gray_raw[mask_row_indices, mask_col_indices].astype(np.float64)
    total_mass = float(pixel_weights.sum())
    if total_mass > 0:
      center_of_mass_x = float((mask_col_indices * pixel_weights).sum() / total_mass)
      center_of_mass_y = float((mask_row_indices * pixel_weights).sum() / total_mass)
    else:
      center_of_mass_x, center_of_mass_y = min_area_rect[0]

  return {
      "box_points": box_corner_points,
      "boundary_points": boundary_points_int,
      "length": length,
      "width": width,
      "avg_intensity": avg_intensity,
      "center_of_mass": (center_of_mass_x, center_of_mass_y),
      "subpixel": refined_ok,
  }


def reset_ui_state_after_video_load():
  """Called right after a successful Load-mode video swap. Everything in
  this module that was computed from (or cached against) the PREVIOUS
  video's frames/dimensions/frame-count needs to be reset, or it'll either
  show stale/wrong data or index out of bounds against the new frames.
  """
  global canvas_w, canvas_h, current_frame, is_playing, mouse_in_frame
  global trail_raw_frame, trail_raw_skip, trail_raw_count, trail_raw_stabilize
  global last_trail_saved_key
  global segment_result, segment_frame_index, segment_message, segment_version
  global latest_original_frame, latest_adjusted_frame

  # The window/canvas width still needs to fit the trackbar labels
  # regardless of the new video's own width; canvas height always tracks
  # the new video's frame height exactly.
  canvas_w = max(frame_w, MIN_WINDOW_W)
  canvas_h = mode_bar_h + frame_h + info_bar_h

  current_frame = -1
  is_playing = False
  # Stale mouse_x/mouse_y from the old video's dimensions could otherwise
  # index outside the new frame's bounds the next time something reads
  # original_frame[mouse_y, mouse_x] -- safest to just require a fresh
  # mouse-move over the new picture before re-enabling the pixel readout.
  mouse_in_frame = False

  # The Trail composite and Segment result both belong to the old video's
  # specific frames; neither makes sense to keep around.
  trail_raw_frame = None
  trail_raw_skip = None
  trail_raw_count = None
  trail_raw_stabilize = None
  last_trail_saved_key = None

  segment_result = None
  segment_frame_index = -1
  segment_message = "Segment mode: click inside a bright object"
  segment_version += 1  # bump so the draw loop notices and clears the overlay

  latest_original_frame = frames_cache[0]
  latest_adjusted_frame = frames_cache[0]

  # The Frame/Trail Skip/Trail Count trackbars' valid ranges depend on the
  # new video's total_frames, so their max needs updating -- and their
  # position reset to a value guaranteed valid for the new range (Trail
  # Count defaults back to "every frame", matching Trail Skip's default).
  try:
    cv2.setTrackbarMax("Frame", "Video Player", total_frames - 1)
    cv2.setTrackbarMax("Trail Skip", "Video Player", max(0, total_frames - 1))
    cv2.setTrackbarMax("Trail Count", "Video Player", total_frames)
  except AttributeError:
    # Older OpenCV builds don't expose setTrackbarMax to Python at all.
    print(
        "Note: this OpenCV build can't resize trackbars at runtime -- "
        "restart the app to get correctly-ranged Frame/Trail Skip/Trail "
        "Count sliders for the newly loaded video."
    )
  cv2.setTrackbarPos("Frame", "Video Player", 0)
  cv2.setTrackbarPos("Trail Skip", "Video Player", 0)
  cv2.setTrackbarPos("Trail Count", "Video Player", total_frames)


def prompt_and_load_new_video():
  """The whole Load action, triggered directly by clicking the LOAD
  button (or pressing 'l') -- no second click on the video needed, since
  unlike Save/Segment/Trail, this action doesn't depend on WHERE you
  click. Opens the file picker, runs the memory-safety check/confirmation,
  loads the chosen file if it goes ahead, and resets everything that
  depended on the previous video. The outcome (success, failure, or
  cancellation) is shown via the same brief "Saved ..."-style flash
  message used elsewhere in the app, so it's visible regardless of
  whatever mode you're in afterward.
  """
  global current_mode, last_save_message, last_save_time, is_playing

  chosen_path = prompt_for_video_path()
  if not chosen_path:
    last_save_message = "Load cancelled (no file chosen)"
    last_save_time = time.time()
    return

  if load_video_file(chosen_path, interactive=True):
    reset_ui_state_after_video_load()
    last_save_message = (
        f"Loaded {os.path.basename(chosen_path)}: "
        f"{total_frames} frames, {frame_w}x{frame_h}"
    )
    print(last_save_message)
    # Drop into View mode on the freshly loaded video, rather than
    # leaving whatever mode was active on the OLD video selected.
    current_mode = MODE_VIEW
    is_playing = False
  else:
    last_save_message = f"Load failed for {os.path.basename(chosen_path)} (see console)"

  last_save_time = time.time()


def enter_play_mode_and_start():
  global current_mode, is_playing, current_frame, last_frame_time
  current_mode = MODE_PLAY
  if current_frame >= total_frames - 1:
    cv2.setTrackbarPos("Frame", "Video Player", 0)
    current_frame = -1
  is_playing = True
  last_frame_time = time.time()


def estimate_frame_to_frame_affine(gray_prev, gray_curr, orb_detector):
  """Estimate the 2x3 affine matrix that maps points in gray_curr into
  gray_prev's coordinate system (i.e. warpAffine(curr_frame, matrix, size)
  aligns the current frame onto the previous one), using ORB features + a
  RANSAC partial-affine fit (translation + rotation + uniform scale -- no
  shear, which suits typical handheld camera motion and is more robust to
  fit than a full homography from sparse matches). Returns None if there
  aren't enough reliable matches to trust a fit, OR if the fit it does
  find is implausible for real adjacent-frame camera motion (see
  is_plausible_frame_to_frame_affine below) -- real-world, especially
  Variable Frame Rate, footage can have noisier/more ambiguous feature
  matches than a clean synthetic test, and an unchecked bad fit here
  doesn't just look a little off: it can warp a whole frame's content to
  a wildly wrong place and visibly wreck the Trail composite.
  """
  keypoints_prev, descriptors_prev = orb_detector.detectAndCompute(gray_prev, None)
  keypoints_curr, descriptors_curr = orb_detector.detectAndCompute(gray_curr, None)
  if (
      descriptors_prev is None or descriptors_curr is None
      or len(keypoints_prev) < 6 or len(keypoints_curr) < 6
  ):
    return None

  # For each feature in the current frame, find its 2 best-matching
  # features in the previous frame (k=2), so a "ratio test" can be applied
  # below to keep only confident, unambiguous matches.
  matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
  candidate_matches = matcher.knnMatch(descriptors_curr, descriptors_prev, k=2)

  good_matches = []
  for match_pair in candidate_matches:
    if len(match_pair) == 2:
      best_match, second_best_match = match_pair
      # Lowe's ratio test: only trust a match if the best candidate is
      # meaningfully closer than the second-best one (otherwise the match
      # is ambiguous and likely to be wrong).
      if best_match.distance < 0.75 * second_best_match.distance:
        good_matches.append(best_match)
  if len(good_matches) < 6:
    return None

  points_in_curr_frame = np.float32(
      [keypoints_curr[m.queryIdx].pt for m in good_matches]
  ).reshape(-1, 1, 2)
  points_in_prev_frame = np.float32(
      [keypoints_prev[m.trainIdx].pt for m in good_matches]
  ).reshape(-1, 1, 2)
  affine_matrix, _inlier_mask = cv2.estimateAffinePartial2D(
      points_in_curr_frame, points_in_prev_frame,
      method=cv2.RANSAC, ransacReprojThreshold=3.0,
  )

  frame_h, frame_w_ = gray_prev.shape[:2]
  if not is_plausible_frame_to_frame_affine(affine_matrix, frame_w_, frame_h):
    return None

  return affine_matrix


def is_plausible_frame_to_frame_affine(matrix, frame_w, frame_h):
  """Sanity-check a frame-to-frame affine fit against what real adjacent-
  frame camera motion could plausibly look like, so a bad/ambiguous
  feature-matching fit gets rejected instead of warping a frame's content
  to a wildly wrong place in the Trail composite.
  """
  if matrix is None:
    return False

  scale_x, shear_y, translate_x = matrix[0]
  shear_x, scale_y, translate_y = matrix[1]

  # estimateAffinePartial2D models a similarity transform (uniform scale +
  # rotation, no independent shear), so the "scale" is the length of
  # either column of the 2x2 linear part -- read from the first column
  # here. A genuine single-frame-step camera zoom is never anywhere close
  # to doubling/halving the image.
  scale = (scale_x ** 2 + shear_x ** 2) ** 0.5
  MIN_PLAUSIBLE_SCALE = 0.5
  MAX_PLAUSIBLE_SCALE = 2.0
  if not (MIN_PLAUSIBLE_SCALE <= scale <= MAX_PLAUSIBLE_SCALE):
    return False

  # A real camera also doesn't jump by more than about half the frame's
  # own size from one frame to the next -- a fit claiming that usually
  # means the feature matches were bogus (e.g. matched onto the wrong
  # part of the scene) rather than describing real motion.
  MAX_TRANSLATION_FRACTION = 0.5
  max_translation = MAX_TRANSLATION_FRACTION * max(frame_w, frame_h)
  if abs(translate_x) > max_translation or abs(translate_y) > max_translation:
    return False

  return True


def get_trail_image(skip_frames, count_frames, stabilize):
  """Return (composite, recomputed) for a given skip_frames/count_frames/
  stabilize combination -- computing it if any of the three has changed
  since the last call (recomputed=True), or returning the cached result
  unchanged otherwise (recomputed=False). Combines
  frames_cache[skip_frames : skip_frames + count_frames] with a per-pixel
  maximum (lighten blend), so anything bright that moved leaves a trail
  against whatever stayed dark and static. E.g. with skip_frames=15,
  count_frames=5, frames 15-19 are combined.

  count_frames matters more than it might look: max-blending EVERY frame
  of an ordinary, well-lit, moving scene tends to wash the whole
  composite out into a smeared mess, since across hundreds of frames
  almost every pixel position ends up bright in at least one of them.
  This technique really shines on a small number of frames with an
  isolated bright subject against a mostly dark, static background
  (fireworks, light trails, star trails) -- narrowing count_frames down
  is the main way to get a clean result on other kinds of footage.

  When stabilize is True, each frame is first aligned back to the first
  frame in that range (via chained frame-to-frame ORB/RANSAC affine
  estimation) before being combined, so camera pan/rotation/zoom doesn't
  smear the background into a ghosted blur -- only genuine motion relative
  to the scene shows up as a trail. Frames where alignment can't be
  estimated confidently (too few feature matches, or a fit that doesn't
  look like plausible real camera motion) fall back to the most-recent-
  known transform rather than aborting.
  """
  global trail_raw_frame, trail_raw_skip, trail_raw_count, trail_raw_stabilize
  skip_frames = max(0, min(skip_frames, total_frames - 1))
  count_frames = max(1, min(count_frames, total_frames - skip_frames))
  if (
      trail_raw_frame is not None
      and trail_raw_skip == skip_frames
      and trail_raw_count == count_frames
      and trail_raw_stabilize == stabilize
  ):
    return trail_raw_frame, False

  frames_to_combine = frames_cache[skip_frames:skip_frames + count_frames]
  composite = frames_to_combine[0].copy()

  if stabilize and len(frames_to_combine) > 1:
    composite_h, composite_w = composite.shape[:2]
    orb_detector = cv2.ORB_create(500)
    previous_frame_gray = cv2.cvtColor(frames_to_combine[0], cv2.COLOR_BGR2GRAY)

    # 3x3 homogeneous transform mapping "this frame's coordinates" into the
    # first (reference) frame's coordinates. Starts as the identity because
    # the first frame trivially maps onto itself.
    cumulative_transform = np.eye(3, dtype=np.float64)

    for raw_frame in frames_to_combine[1:]:
      current_frame_gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
      frame_to_frame_affine = estimate_frame_to_frame_affine(
          previous_frame_gray, current_frame_gray, orb_detector
      )
      frame_to_frame_homogeneous = np.eye(3, dtype=np.float64)
      if frame_to_frame_affine is not None:
        frame_to_frame_homogeneous[:2, :] = frame_to_frame_affine
      # else: no confident (or plausible) estimate for this frame -- treat
      # as unchanged from the previous frame's cumulative transform rather
      # than aborting the whole composite.

      # Chain this frame's transform onto the running total, so every
      # frame ends up expressed directly in the reference frame's
      # coordinate system (not just relative to its immediate neighbor).
      cumulative_transform = cumulative_transform @ frame_to_frame_homogeneous
      warp_matrix_2x3 = cumulative_transform[:2, :].astype(np.float32)
      aligned_frame = cv2.warpAffine(raw_frame, warp_matrix_2x3, (composite_w, composite_h))
      np.maximum(composite, aligned_frame, out=composite)
      previous_frame_gray = current_frame_gray
  else:
    for raw_frame in frames_to_combine[1:]:
      np.maximum(composite, raw_frame, out=composite)

  trail_raw_frame = composite
  trail_raw_skip = skip_frames
  trail_raw_count = count_frames
  trail_raw_stabilize = stabilize
  return trail_raw_frame, True


def save_trail_image(raw_composite, brightness_val, contrast_val):
  """Apply the given brightness/contrast to a trail composite and save it to
  disk, with a filename that records those slider values. Returns the
  filename (not the full path) for use in confirmation messages.
  """
  alpha = contrast_val / 10.0
  beta = brightness_val - 50
  adjusted = cv2.convertScaleAbs(raw_composite, alpha=alpha, beta=beta)
  os.makedirs(TRAIL_DIR, exist_ok=True)
  filename = f"trail_brightness{brightness_val:03d}_contrast{contrast_val:03d}.png"
  filepath = os.path.join(TRAIL_DIR, filename)
  cv2.imwrite(filepath, adjusted)
  return filename


def mouse_callback(event, x, y, flags, param):
  global is_playing, last_frame_time, mouse_x, mouse_y, mouse_in_frame
  global current_frame, current_mode, save_counter, last_save_message
  global last_save_time, segment_result, segment_frame_index
  global segment_message, segment_version, last_trail_saved_key, show_help

  # (x, y) from OpenCV are in ON-SCREEN WINDOW pixels, which is NOT the
  # same thing as pixels in `canvas` if the user has resized the window
  # (OpenCV stretches/shrinks the displayed image to fit). getWindowImageRect
  # gives the window's current on-screen size, so we can work out the
  # scale factor and convert the click back into canvas coordinates.
  window_rect = cv2.getWindowImageRect("Video Player")
  if window_rect[2] > 0 and window_rect[3] > 0:
    scale_x = canvas_w / window_rect[2]
    scale_y = canvas_h / window_rect[3]
    actual_x = int(x * scale_x)
    actual_y = int(y * scale_y)
  else:
    actual_x, actual_y = x, y

  # actual_x/actual_y are now in CANVAS coordinates (top-left of the whole
  # window, including the mode bar). Mode/help buttons live in that same
  # coordinate space, so they're checked directly against actual_x/actual_y
  # further down. Everything involving the video picture itself, though,
  # needs FRAME-LOCAL coordinates -- i.e. (0, 0) at the video's own
  # top-left corner -- which just means subtracting the mode bar's height
  # from the y coordinate (x doesn't need adjusting since the frame is
  # left-aligned at canvas x=0).
  frame_x = actual_x
  frame_y = actual_y - frame_y_offset

  if event == cv2.EVENT_MOUSEMOVE:
    mouse_x, mouse_y = frame_x, frame_y
    # Only treat the cursor as "in frame" while it's over the video image
    # itself, not the mode bar above it or the info bar below it.
    mouse_in_frame = 0 <= frame_x < frame_w and 0 <= frame_y < frame_h

  elif event == cv2.EVENT_LBUTTONDOWN:
    # The help overlay swallows every click while it's showing: any click
    # just dismisses it (rather than also acting on whatever's underneath),
    # so a second click is needed to actually act on a button.
    if show_help:
      show_help = False
      return

    if help_btn_x1 <= actual_x <= help_btn_x2 and help_btn_y1 <= actual_y <= help_btn_y2:
      show_help = True
      return

    if load_btn_x1 <= actual_x <= load_btn_x2 and load_btn_y1 <= actual_y <= load_btn_y2:
      prompt_and_load_new_video()
      return

    # Mode buttons live in the bar ABOVE the video image, so they're
    # checked against canvas-level coordinates, not frame-local ones, and
    # they take priority over everything else.
    if view_btn_x1 <= actual_x <= view_btn_x2 and view_btn_y1 <= actual_y <= view_btn_y2:
      current_mode = MODE_VIEW
      is_playing = False
      return

    if save_btn_x1 <= actual_x <= save_btn_x2 and save_btn_y1 <= actual_y <= save_btn_y2:
      current_mode = MODE_SAVE
      is_playing = False  # pause so the frame being captured is predictable
      return

    if play_btn_x1 <= actual_x <= play_btn_x2 and play_btn_y1 <= actual_y <= play_btn_y2:
      if current_mode != MODE_PLAY:
        # Entering Play mode always (re)starts playback.
        enter_play_mode_and_start()
      else:
        # Already in Play mode: this click toggles play/pause.
        is_playing = not is_playing
        if is_playing:
          if current_frame >= total_frames - 1:
            cv2.setTrackbarPos("Frame", "Video Player", 0)
            current_frame = -1
          last_frame_time = time.time()
      return

    if segment_btn_x1 <= actual_x <= segment_btn_x2 and segment_btn_y1 <= actual_y <= segment_btn_y2:
      current_mode = MODE_SEGMENT
      is_playing = False  # pause so the object being segmented is stable
      return

    if trail_btn_x1 <= actual_x <= trail_btn_x2 and trail_btn_y1 <= actual_y <= trail_btn_y2:
      current_mode = MODE_TRAIL
      is_playing = False
      return

    # Everything below this point happens inside the video frame itself, so
    # work in frame-local coordinates.
    if not (0 <= frame_x < frame_w and 0 <= frame_y < frame_h):
      return

    # In Image Save mode, a click anywhere on the video frame captures a
    # CAPTURE_SIZE x CAPTURE_SIZE crop centered on the click, taken from the
    # currently displayed (brightness/contrast-adjusted) frame, without any
    # button/text overlays baked in.
    if current_mode == MODE_SAVE:
      filename = save_crop(latest_adjusted_frame, frame_x, frame_y, CAPTURE_SIZE)
      save_counter += 1
      last_save_message = f"Saved {filename}"
      last_save_time = time.time()
      print(last_save_message)

    # In Segment mode, a click inside a bright object finds its boundary
    # against the RAW frame (unaffected by the brightness/contrast sliders)
    # and reports its length, width, and average intensity from that same
    # raw data.
    elif current_mode == MODE_SEGMENT:
      gray = cv2.cvtColor(latest_original_frame, cv2.COLOR_BGR2GRAY)
      result = segment_bright_object(gray, frame_x, frame_y)
      segment_frame_index = current_frame
      if result is None:
        segment_result = None
        segment_message = "No suitably-sized bright object found at that click point"
      else:
        segment_result = result
        precision_note = "" if result["subpixel"] else " (coarse boundary)"
        com_x, com_y = result["center_of_mass"]
        segment_message = (
            f"Frame {segment_frame_index}  |  "
            f"Length: {result['length']:.2f}px  Width: {result['width']:.2f}px  "
            f"Avg Intensity: {result['avg_intensity']:.1f}  |  "
            f"Center of Mass: ({com_x:.1f}, {com_y:.1f}){precision_note}"
        )
        print(segment_message)
      segment_version += 1

    # In Trail mode, a click anywhere on the video frame saves the current
    # trail composite (with the CURRENT brightness/contrast/skip/stabilize
    # settings applied) to disk -- but only if something has actually
    # changed since the last save, so repeated clicks with unchanged
    # settings don't write identical duplicate files. (The composite is
    # also auto-saved once immediately whenever it's freshly (re)computed,
    # in the main loop -- this covers the case of tweaking
    # brightness/contrast afterward without changing skip/stabilize, which
    # doesn't trigger a recompute on its own.)
    elif current_mode == MODE_TRAIL:
      brightness_now = cv2.getTrackbarPos("Brightness", "Video Player")
      contrast_now = cv2.getTrackbarPos("Contrast", "Video Player")
      skip_now = cv2.getTrackbarPos("Trail Skip", "Video Player")
      count_now = cv2.getTrackbarPos("Trail Count", "Video Player")
      stabilize_now = bool(cv2.getTrackbarPos("Stabilize", "Video Player"))
      current_key = (skip_now, count_now, stabilize_now, brightness_now, contrast_now)
      if current_key == last_trail_saved_key:
        pass  # nothing has changed since the last save; don't duplicate it
      else:
        raw_composite, _ = get_trail_image(skip_now, count_now, stabilize_now)
        filename = save_trail_image(raw_composite, brightness_now, contrast_now)
        last_trail_saved_key = current_key
        last_save_message = f"Saved {filename}"
        last_save_time = time.time()
        print(last_save_message)

cv2.setMouseCallback("Video Player", mouse_callback)

while True:
  # Read every trackbar's current position once per loop iteration. These
  # are also compared against the `prev_*` values below (set at the very
  # end of this loop, after the last frame that used them) to figure out
  # whether anything actually changed since the last redraw.
  target_frame = cv2.getTrackbarPos("Frame", "Video Player")
  brightness_val = cv2.getTrackbarPos("Brightness", "Video Player")
  contrast_val = cv2.getTrackbarPos("Contrast", "Video Player")
  trail_skip_val = cv2.getTrackbarPos("Trail Skip", "Video Player")
  trail_count_val = cv2.getTrackbarPos("Trail Count", "Video Player")
  stabilize_val = cv2.getTrackbarPos("Stabilize", "Video Player")

  if is_playing:
    # Advance the Frame trackbar based on real elapsed time rather than
    # "one frame per loop iteration", so playback speed matches the
    # source video's actual frame rate regardless of how fast this loop
    # happens to run.
    current_time = time.time()
    elapsed = current_time - last_frame_time

    if elapsed >= frame_interval:
      frames_to_advance = int(elapsed // frame_interval)
      target_frame = current_frame + frames_to_advance

      if target_frame >= total_frames:
        target_frame = total_frames - 1
        is_playing = False

      cv2.setTrackbarPos("Frame", "Video Player", target_frame)
      # Carry over the leftover (sub-frame-interval) time instead of
      # resetting the clock to now, so timing doesn't drift over a long
      # playback session.
      last_frame_time = current_time - (elapsed % frame_interval)

  # Figure out whether ANYTHING changed since the picture currently on
  # screen was drawn. Redrawing every single loop iteration regardless
  # would waste CPU on expensive operations (especially the per-pixel
  # brightness/contrast adjustment and, in Segment/Trail modes, real
  # image-processing work) for no visible benefit when nothing changed.
  image_settings_changed = (brightness_val != prev_brightness) or (
      contrast_val != prev_contrast
  )
  mode_changed = current_mode != prev_mode
  playing_state_changed = is_playing != prev_is_playing
  save_message_active = (time.time() - last_save_time) < SAVE_MESSAGE_DURATION_S
  segment_updated = segment_version != prev_segment_version
  trail_skip_changed = trail_skip_val != prev_trail_skip
  trail_count_changed = trail_count_val != prev_trail_count
  stabilize_changed = stabilize_val != prev_stabilize
  help_changed = show_help != prev_show_help

  # Redraw whenever the frame/settings/mode/play-state change, during
  # playback, when the mouse moves (RGB readout / crop-preview overlay
  # needs to track the cursor), when a new segmentation result comes in, or
  # while a "Saved ..." confirmation message is showing.
  if (
      current_frame != target_frame
      or image_settings_changed
      or mode_changed
      or playing_state_changed
      or is_playing
      or mouse_in_frame
      or save_message_active
      or segment_updated
      or trail_skip_changed
      or trail_count_changed
      or stabilize_changed
      or help_changed
  ):
    if current_mode == MODE_TRAIL:
      original_frame, trail_recomputed = get_trail_image(trail_skip_val, trail_count_val, bool(stabilize_val))
      if trail_recomputed:
        # Auto-save immediately whenever the underlying composite is freshly
        # computed (first entry into Trail mode, or a Trail Skip/Trail
        # Count/Stabilize change), using whatever brightness/contrast are
        # set right now.
        filename = save_trail_image(original_frame, brightness_val, contrast_val)
        last_trail_saved_key = (trail_skip_val, trail_count_val, bool(stabilize_val), brightness_val, contrast_val)
        last_save_message = f"Saved {filename}"
        last_save_time = time.time()
        print(last_save_message)
    else:
      original_frame = frames_cache[target_frame]

    alpha = contrast_val / 10.0
    beta = brightness_val - 50
    adjusted_frame = cv2.convertScaleAbs(original_frame, alpha=alpha, beta=beta)

    latest_original_frame = original_frame
    latest_adjusted_frame = adjusted_frame

    display_frame = adjusted_frame.copy()

    save_active = current_mode == MODE_SAVE
    view_active = current_mode == MODE_VIEW
    play_active = current_mode == MODE_PLAY
    segment_active = current_mode == MODE_SEGMENT
    trail_active = current_mode == MODE_TRAIL

    # --- Image Save mode: preview the capture region under the cursor ---
    if save_active and mouse_in_frame:
      half = CAPTURE_SIZE // 2
      prev_x1 = mouse_x - half
      prev_y1 = mouse_y - half
      prev_x2 = mouse_x + half
      prev_y2 = mouse_y + half
      cv2.rectangle(
          display_frame, (prev_x1, prev_y1), (prev_x2, prev_y2), (0, 140, 255), 2
      )

    # --- Segment mode: draw the actual refined sub-pixel boundary of the
    # last-found object (not just its bounding box), as long as it was
    # found on the frame currently on screen ---
    segment_valid_for_this_frame = (
        segment_active and segment_result is not None and segment_frame_index == target_frame
    )
    if segment_valid_for_this_frame:
      cv2.polylines(
          display_frame,
          [segment_result["boundary_points"]],
          isClosed=True,
          color=(0, 255, 255),
          thickness=1,
          lineType=cv2.LINE_AA,
      )

      # Target reticle at the center of mass: four short tick marks (up,
      # down, left, right) that start clear of the object and point inward
      # toward it, rather than a crosshair drawn across the object itself.
      com_x, com_y = segment_result["center_of_mass"]
      com_x_i, com_y_i = int(round(com_x)), int(round(com_y))

      obj_length = segment_result["length"]
      obj_width = segment_result["width"]
      # Half the object's bounding diagonal, so the gap clears the object
      # regardless of its rotation.
      gap_radius = int(round(((obj_length ** 2 + obj_width ** 2) ** 0.5) / 2.0)) + 6
      tick_length = 14
      outer_radius = gap_radius + tick_length
      reticle_color = (0, 255, 255)

      cv2.line(display_frame, (com_x_i, com_y_i - outer_radius), (com_x_i, com_y_i - gap_radius), reticle_color, 1, cv2.LINE_AA)
      cv2.line(display_frame, (com_x_i, com_y_i + gap_radius), (com_x_i, com_y_i + outer_radius), reticle_color, 1, cv2.LINE_AA)
      cv2.line(display_frame, (com_x_i - outer_radius, com_y_i), (com_x_i - gap_radius, com_y_i), reticle_color, 1, cv2.LINE_AA)
      cv2.line(display_frame, (com_x_i + gap_radius, com_y_i), (com_x_i + outer_radius, com_y_i), reticle_color, 1, cv2.LINE_AA)

    # Build the full canvas: a mode-button bar on top, the (possibly
    # brightness/contrast adjusted) video frame with its own overlays in
    # the middle, and the RGB-readout info bar at the bottom. Both bars are
    # kept fully off the video image itself so nothing ever obscures the
    # picture.
    canvas = np.full((canvas_h, canvas_w, 3), 30, dtype="uint8")
    canvas[frame_y_offset:frame_y_offset + frame_h, 0:frame_w] = display_frame

    # --- Mode bar (top strip) ---
    # LOAD is a one-shot action button, not a lingering mode, so it's
    # always drawn in the neutral color -- never "active".
    load_color = (70, 70, 70)
    view_color = (0, 140, 255) if view_active else (70, 70, 70)
    save_color = (0, 140, 255) if save_active else (70, 70, 70)
    segment_color = (0, 140, 255) if segment_active else (70, 70, 70)
    trail_color = (0, 140, 255) if trail_active else (70, 70, 70)
    help_color = (0, 140, 255) if show_help else (70, 70, 70)
    if play_active:
      play_color = (0, 200, 0) if is_playing else (0, 0, 255)
    else:
      play_color = (70, 70, 70)

    cv2.rectangle(canvas, (load_btn_x1, load_btn_y1), (load_btn_x2, load_btn_y2), load_color, -1)
    cv2.rectangle(canvas, (view_btn_x1, view_btn_y1), (view_btn_x2, view_btn_y2), view_color, -1)
    cv2.rectangle(canvas, (save_btn_x1, save_btn_y1), (save_btn_x2, save_btn_y2), save_color, -1)
    cv2.rectangle(canvas, (play_btn_x1, play_btn_y1), (play_btn_x2, play_btn_y2), play_color, -1)
    cv2.rectangle(canvas, (segment_btn_x1, segment_btn_y1), (segment_btn_x2, segment_btn_y2), segment_color, -1)
    cv2.rectangle(canvas, (trail_btn_x1, trail_btn_y1), (trail_btn_x2, trail_btn_y2), trail_color, -1)
    cv2.rectangle(canvas, (help_btn_x1, help_btn_y1), (help_btn_x2, help_btn_y2), help_color, -1)

    cv2.putText(
        canvas, "LOAD", (load_btn_x1 + 22, load_btn_y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
    )
    cv2.putText(
        canvas, "VIEW", (view_btn_x1 + 22, view_btn_y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
    )
    cv2.putText(
        canvas, "SAVE", (save_btn_x1 + 22, save_btn_y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
    )
    play_label = "PAUSE" if (play_active and is_playing) else "PLAY"
    play_label_x = play_btn_x1 + (12 if play_label == "PAUSE" else 24)
    cv2.putText(
        canvas, play_label, (play_label_x, play_btn_y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
    )
    cv2.putText(
        canvas, "SEGMENT", (segment_btn_x1 + 6, segment_btn_y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2,
    )
    cv2.putText(
        canvas, "TRAIL", (trail_btn_x1 + 20, trail_btn_y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
    )
    cv2.putText(
        canvas, "HELP", (help_btn_x1 + 24, help_btn_y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
    )

    # --- Info bar (bottom strip) ---
    if mouse_in_frame:
      b, g, r = original_frame[mouse_y, mouse_x]
      avg = (int(r) + int(g) + int(b)) / 3.0
      if save_active:
        readout_text = (
            f"Pixel ({mouse_x}, {mouse_y})  R:{r}  G:{g}  B:{b}  "
            f"Avg:{avg:.1f}  |  Click to save a {CAPTURE_SIZE}x{CAPTURE_SIZE} crop"
        )
      elif segment_active:
        readout_text = segment_message
      elif trail_active:
        stab_tag = " [stabilized]" if stabilize_val else ""
        trail_end = min(trail_skip_val + trail_count_val, total_frames) - 1
        readout_text = (
            f"Pixel ({mouse_x}, {mouse_y})  R:{r}  G:{g}  B:{b}  Avg:{avg:.1f}  |  "
            f"Frames {trail_skip_val}-{trail_end}{stab_tag}  |  Click to save the trail image"
        )
      else:
        readout_text = (
            f"Pixel ({mouse_x}, {mouse_y})  R:{r}  G:{g}  B:{b}  Avg:{avg:.1f}"
        )
    else:
      b, g, r = 0, 0, 0
      if save_active:
        readout_text = "Image Save mode: click on the video to save a crop"
      elif play_active:
        readout_text = "Playing..." if is_playing else "Paused"
      elif segment_active:
        readout_text = segment_message
      elif trail_active:
        stab_tag = " [stabilized]" if stabilize_val else ""
        trail_end = min(trail_skip_val + trail_count_val, total_frames) - 1
        readout_text = (
            f"Trail mode (frames {trail_skip_val}-{trail_end}{stab_tag}): "
            f"click on the video to save the combined-frame trail image"
        )
      else:
        readout_text = "Move the mouse over the video to inspect a pixel"

    if save_message_active:
      readout_text = last_save_message

    info_bar_y = frame_y_offset + frame_h
    text_y = info_bar_y + info_bar_h // 2 + 6
    cv2.putText(
        canvas,
        readout_text,
        (12, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0) if save_message_active else (255, 255, 255),
        2,
    )

    # Small color swatch showing the original pixel color, at the right
    # edge of the info bar (View / Image Save modes only, and not while a
    # save confirmation message is showing).
    if mouse_in_frame and not save_message_active and not segment_active:
      swatch_size = info_bar_h - 16
      swatch_x2 = canvas_w - 12
      swatch_x1 = swatch_x2 - swatch_size
      swatch_y1 = info_bar_y + 8
      swatch_y2 = swatch_y1 + swatch_size
      cv2.rectangle(
          canvas,
          (swatch_x1, swatch_y1),
          (swatch_x2, swatch_y2),
          (int(b), int(g), int(r)),
          -1,
      )
      cv2.rectangle(
          canvas,
          (swatch_x1, swatch_y1),
          (swatch_x2, swatch_y2),
          (255, 255, 255),
          1,
      )

    # --- Help overlay: dismissable, drawn on top of everything else ---
    if show_help:
      overlay = canvas.copy()
      cv2.rectangle(overlay, (0, 0), (canvas_w, canvas_h), (0, 0, 0), -1)
      canvas = cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0)

      # Kept short and punchy on purpose: this text needs to read as large
      # as possible, which means as few lines as possible while still
      # covering every mode/control/behavior.
      help_lines = [
          "HELP (click anywhere or press h to close)",
          "",
          "LOAD: click to choose a video file to open (warns first",
          "  if it looks too big to fit comfortably in memory)",
          "VIEW: scrub Frame slider, hover for pixel RGB",
          "SAVE: click to save a 300x300 crop at the click point",
          "PLAY: click to start playback, click again to pause",
          "",
          "SEGMENT: click a bright object for its sub-pixel boundary,",
          "  length, width, avg intensity, center of mass",
          "",
          "TRAIL: max-blends frames into one motion-trail image",
          "  Trail Skip = first frame used, Trail Count = how many",
          "  (fewer frames avoids a washed-out result on ordinary",
          "  footage), Stabilize = ORB align",
          "  Auto-saves when Trail Skip/Count/Stabilize changes the",
          "  image. Brightness/Contrast changes do NOT auto-save --",
          "  click the video to save that look (skips duplicate",
          "  saves if nothing changed since the last one)",
          "  -> ./trails/trail_brightness###_contrast###.png",
          "",
          "Sliders: Frame/Brightness/Contrast affect every mode;",
          "  Trail Skip/Count/Stabilize affect Trail only",
          "",
          "Keys: l v s p g t h = modes, space = play/pause,",
          "  arrows = step frame, q = quit",
      ]

      text_x = 20
      top_margin = 24
      bottom_margin = 16
      num_lines = len(help_lines)

      # Size the text to the available space rather than a small fixed
      # size: as large as the frame allows, only shrinking as far as
      # necessary to still fit every line on an unusually short one.
      avail_h = canvas_h - top_margin - bottom_margin
      line_h = max(28, min(64, avail_h / num_lines))
      font_scale = max(0.85, min(2.0, line_h / 24.0))
      thickness = 2 if font_scale < 1.1 else 3

      # Shrink further if the longest line would run off the window's
      # width at this size.
      longest = max(help_lines, key=len)
      (text_w, _), _ = cv2.getTextSize(longest, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
      max_text_w = canvas_w - 2 * text_x
      if text_w > max_text_w:
        font_scale *= max_text_w / text_w
        thickness = 2 if font_scale < 1.1 else 3

      text_y = top_margin + int(line_h * 0.75)
      for line in help_lines:
        if text_y > canvas_h - bottom_margin // 2:
          break  # ran out of vertical room; rest is truncated
        if line:
          cv2.putText(
              canvas, line, (text_x, text_y),
              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
          )
        text_y += int(line_h)

    cv2.imshow("Video Player", canvas)

    current_frame = target_frame
    prev_brightness = brightness_val
    prev_contrast = contrast_val
    prev_mode = current_mode
    prev_is_playing = is_playing
    prev_segment_version = segment_version
    prev_trail_skip = trail_skip_val
    prev_trail_count = trail_count_val
    prev_stabilize = stabilize_val
    prev_show_help = show_help

  key = cv2.waitKeyEx(1)

  if (key & 0xFF) == ord("q"):
    break
  elif (key & 0xFF) == ord(" "):
    if current_mode != MODE_PLAY:
      enter_play_mode_and_start()
    else:
      is_playing = not is_playing
      if is_playing:
        if current_frame >= total_frames - 1:
          cv2.setTrackbarPos("Frame", "Video Player", 0)
          current_frame = -1
        last_frame_time = time.time()
  elif (key & 0xFF) == ord("v"):
    current_mode = MODE_VIEW
    is_playing = False
  elif (key & 0xFF) == ord("s"):
    current_mode = MODE_SAVE
    is_playing = False
  elif (key & 0xFF) == ord("p"):
    enter_play_mode_and_start()
  elif (key & 0xFF) == ord("g"):
    current_mode = MODE_SEGMENT
    is_playing = False
  elif (key & 0xFF) == ord("t"):
    current_mode = MODE_TRAIL
    is_playing = False
  elif (key & 0xFF) == ord("l"):
    prompt_and_load_new_video()
  elif (key & 0xFF) == ord("h"):
    show_help = not show_help
  elif key == 63235:
    # Right arrow. NOTE: 63235/63234 are the raw key codes OpenCV's
    # macOS (Cocoa) backend reports for the arrow keys -- these will NOT
    # match on Windows/Linux, where the codes are different. Step forward
    # one frame and pause (so stepping doesn't fight with autoplay).
    is_playing = False
    next_frame = min(current_frame + 1, total_frames - 1)
    cv2.setTrackbarPos("Frame", "Video Player", next_frame)
  elif key == 63234:
    # Left arrow (macOS-only code, see note above). Step back one frame.
    is_playing = False
    prev_frame = max(current_frame - 1, 0)
    cv2.setTrackbarPos("Frame", "Video Player", prev_frame)

cv2.destroyAllWindows()
