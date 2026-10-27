# Getting started as an annotator

Lithium is the point-cloud labeller we use to mark vertebral surfaces
(endplates, rims, pedicles, processes) on 3-D bone models. You get a
project bundle (`.li`) with the bones already inside; you label them
and send a small bundle of labels back.

## 1. Install (once, ~5 minutes)

You need Git. On Windows, [GitHub Desktop](https://desktop.github.com)
is the simplest way to clone; on macOS/Linux use `git` in a terminal.

1. Clone the repository you were given access to
   (`github.com/saffffari/Lithium`) to a folder of your choice.
2. Open a terminal in that folder and run the setup script. It
   installs a private Python, the packages and launches the app:

   - **Windows (PowerShell):** `.\setup.ps1`
     If Windows refuses to run scripts:
     `powershell -ExecutionPolicy Bypass -File .\setup.ps1`
   - **macOS / Linux:** `./setup.sh`

   After the first time, start Lithium with `run.bat` (Windows) or
   `./run.sh`.

The app needs an OpenGL 4.3 GPU — any laptop from the last ten years
with up-to-date graphics drivers is fine. No NVIDIA card is required.

## 2. Load the project

Download the `.li` file from the shared Drive folder, then in Lithium:
**File > Import Project (.li)...** and pick it. A progress ring runs
for a minute; the project then opens in CONTACT SHEETS with every bone
as a tile. (The setup script can do this too: `.\setup.ps1 -Kit path\to\file.li`.)

## 3. Label

Press `?` inside the app for the shortcut card; its GETTING STARTED
block is the whole loop:

1. Double-click a bone tile to open it in the LIGHT TABLE.
2. Select points with the box (`B`), lasso (`O`) or brush (`K`).
3. `G` cycles the active label; `Enter` applies it to the selection.
4. Thumb wheel or `LEFT`/`RIGHT` moves to the next bone. Labels save
   themselves.
5. `Ctrl+Z` undoes. `SPACE` refits the view if you get lost.

Your labelling protocol (which structures, which bones, in which
order) comes separately from Alex. Follow it exactly: the study
compares your labels with other readers', so consistency matters more
than speed.

## 4. Send labels back

**File > Export Project (.li)...**, tick **Labels** only, save the
file (a few MB) and drop it in the shared Drive folder with your name
and the date in the filename, e.g. `kevin_2026-11-03_labels.li`.
Importing it on our side merges your labels into the source project
as their own layer.

## If something breaks

- `setup` says `uv` is not on PATH: close the terminal, open a new
  one, run the script again.
- Black window / crash on launch: update your graphics driver.
- File dialogs don't open (Linux): install `python3-tk`.
- Anything else: screenshot + the text in the terminal, send it over.
