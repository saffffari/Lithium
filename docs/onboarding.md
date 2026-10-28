# Labelling a shared project

Someone has sent you a Lithium project as a `.li` file. It holds the point
clouds and the list of labels to use. You install Lithium, import the file,
paint the labels, and send back a small file that holds only your labels.

This page is the short version. The [user guide](user_guide.md) has
everything else.

## 1. Install (once)

You need:

- **Git.** On Windows, [GitHub Desktop](https://desktop.github.com) is the
  easiest way to clone. On macOS and Linux, use `git` in a terminal.
- **A C++ compiler**, for one package that builds during setup:
  - Windows: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/),
    "Desktop development with C++" workload
  - macOS: `xcode-select --install`
  - Linux: `g++` from your package manager
- **A graphics driver with OpenGL 4.1.** Most computers from the last ten
  years have one once the driver is up to date. You do not need an NVIDIA
  card.

Then:

1. Clone `https://github.com/saffffari/Lithium` to a folder of your choice.
2. Open a terminal in that folder and run the setup script. It installs a
   private Python and the packages, then starts Lithium.
   - **Windows (PowerShell):** `.\setup.ps1`
     If Windows refuses to run scripts:
     `powershell -ExecutionPolicy Bypass -File .\setup.ps1`
   - **macOS / Linux:** `./setup.sh`

From then on, start Lithium with `run.bat` (Windows) or `./run.sh`.

## 2. Import the project

In Lithium, choose **File > Import Project (.li)...** and pick the file. A
progress card shows the import and closes itself a few seconds after it finishes. The project
opens with every cloud as a tile in the gallery.

You can also import while installing: `.\setup.ps1 -Kit path\to\file.li` on
Windows, or `./setup.sh --kit path/to/file.li` on macOS and Linux.

## 3. Label

Press `?` in the app for the shortcut card. The loop is:

1. **Double-click** a tile to open the cloud in the **Light Table**.
2. **Click a label** in the LABELS list, or press `G` to step through them.
3. **Pick a tool:** pick (`P`), lasso (`O`), brush (`K`), curve (`U`), or box
   and polygon from the toolbar at the top left of the view.
4. **Hold Shift and drag** to paint the label (Shift+click for pick).
   **Hold Ctrl and drag** to erase it. A plain left-drag pans the view;
   right-drag turns it.
5. **Move to the next cloud** with `V` or `Right`, the previous one with `B`
   or `Left`, or a mouse thumb wheel.

Labels save themselves after every stroke. There is no Save button.
`Ctrl+Z` undoes. `F` refits the view if you get lost.

Whoever sent you the project should also tell you what to label and how.
Follow those instructions closely. Consistency matters more than speed.

## 4. Send your labels back

1. Choose **File > Export Project (.li)...**.
2. Pick the project and tick **Labels** only.
3. Click **Export...** and save the file. It is usually a few MB.
4. Put your name in the file name, for example
   `<project>_<your-name>_labels.li`, and send it back the way you agreed.

When the sender imports your file, your labels arrive as their own project
next to the original, so they never mix with anyone else's.

## If something breaks

- **Setup fails while building `imgui`:** install the C++ compiler above and
  run the setup script again.
- **Setup says `uv` is not on PATH:** close the terminal, open a new one, and
  run the script again.
- **Black window or crash on launch:** update your graphics driver.
- **File dialogs don't open (Linux):** install `python3-tk`.
- **Anything else:** take a screenshot, copy the text from the terminal, and
  send both to whoever sent you the project.
