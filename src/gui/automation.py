"""Automation CLI engine for Lithium — interactive command interface."""

import os
import shlex

from src.core.modes import (
    MODE_CONTACT_SHEETS, MODE_LIGHT_TABLE, MODE_AUTOMATION,
)


class CLIEngine:
    """Command-line interface for bulk operations within the GUI."""

    def __init__(self, app):
        self.app = app
        self.output: list[tuple[str, str]] = []  # (text, level)
        self.history: list[str] = []
        self.history_index: int = -1
        self.input_buf: str = ""
        self._commands: dict = {}
        # Set by log() whenever new output is appended; consumed by the
        # panel draw loop to force a scroll-to-bottom on the next frame,
        # which avoids the stale scroll_max_y race.
        self._pending_scroll: bool = False
        self._register_commands()
        self.log("Lithium Automation Console", "success")
        self.log("Type 'help' for available commands.", "info")

    def _register_commands(self):
        self._commands = {
            'help': self._cmd_help,
            'load': self._cmd_load,
            'load-recursive': self._cmd_load_recursive,
            'list': self._cmd_list,
            'select': self._cmd_select,
            'unload': self._cmd_unload,
            'info': self._cmd_info,
            'export': self._cmd_export,
            'spin': self._cmd_spin,
            'batch-export': self._cmd_batch_export,
            'export-dataset': self._cmd_export_dataset,
            'label': self._cmd_label,
            'labels': self._cmd_labels,
            'propagate': self._cmd_propagate,
            'propagate-all': self._cmd_propagate_all,
            'save-project': self._cmd_save_project,
            'load-project': self._cmd_load_project,
            'train': self._cmd_train,
            'train-status': self._cmd_train_status,
            'train-stop': self._cmd_train_stop,
            'set': self._cmd_set,
            'get': self._cmd_get,
            'camera': self._cmd_camera,
            'clear': self._cmd_clear,
            'status': self._cmd_status,
            'wipe-catalog': self._cmd_wipe_catalog,
            'catalog-status': self._cmd_catalog_status,
        }

    def log(self, text: str, level: str = "info"):
        self.output.append((text, level))
        if len(self.output) > 1000:
            self.output = self.output[-800:]
        self._pending_scroll = True

    def execute(self, line: str):
        line = line.strip()
        if not line:
            return
        self.history.append(line)
        self.history_index = -1
        self.log(f"> {line}", "cmd")

        try:
            parts = shlex.split(line)
        except ValueError:
            parts = line.split()

        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in self._commands:
            try:
                self._commands[cmd](args)
            except Exception as e:
                self.log(f"Error: {e}", "error")
        else:
            self.log(f"Unknown command: '{cmd}'. Type 'help' for a list.", "error")

    def history_up(self) -> str:
        if not self.history:
            return self.input_buf
        if self.history_index == -1:
            self.history_index = len(self.history) - 1
        elif self.history_index > 0:
            self.history_index -= 1
        return self.history[self.history_index]

    def history_down(self) -> str:
        if not self.history or self.history_index == -1:
            return ""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            return self.history[self.history_index]
        else:
            self.history_index = -1
            return ""

    # --- Commands ---

    def _cmd_help(self, args):
        commands = {
            'help': 'Show this help message',
            'load <path>': 'Load a file or directory',
            'list': 'List loaded clouds',
            'select <index>': 'Select a cloud by index',
            'unload <index|all>': 'Remove cloud(s)',
            'info [index]': 'Show cloud details',
            'export <path> [--res WxH]': 'Export screenshot',
            'spin <dir> [--frames N] [--res WxH]': 'Spin render',
            'batch-export <dir> [--res WxH]': 'Export all clouds',
            'export-dataset <dir> [--format ptv3|npz|h5] [--split t,v,ts]': 'Export labeled dataset',
            'label <name> [#rrggbb]': 'Create a new label',
            'labels': 'List all labels with point counts',
            'propagate [--radius R] [--k K]': 'Propagate labels to next frame (4D)',
            'propagate-all [--radius R]': 'Propagate labels through all remaining frames',
            'save-project <path>': 'Save annotation project',
            'load-project <path>': 'Load annotation project',
            'train <data_dir> [--epochs N] [--lr L] [--device d]': 'Launch training sidecar',
            'train-status': 'Show training progress',
            'train-stop': 'Stop training',
            'set <key> <value>': 'Set parameter',
            'get <key>': 'Get parameter value',
            'camera <preset>': 'Set camera preset',
            'clear': 'Clear output',
            'status': 'Show app status',
        }
        if args:
            cmd = args[0].lower()
            for name, desc in commands.items():
                if name.startswith(cmd):
                    self.log(f"  {name}  --  {desc}", "info")
                    return
            self.log(f"No help for '{cmd}'", "error")
        else:
            self.log("Available commands:", "info")
            for name, desc in commands.items():
                self.log(f"  {name}", "info")

    def _cmd_load(self, args):
        if not args:
            self.log("Usage: load <path>", "error")
            return
        path = args[0]
        if not os.path.exists(path):
            self.log(f"Path not found: {path}", "error")
            return
        before = len(self.app.entries)
        self.app.import_path(path)
        after = len(self.app.entries)
        loaded = after - before
        if loaded > 0:
            self.log(f"Loaded {loaded} cloud(s) from {os.path.basename(path)}", "success")
        else:
            self.log(f"No clouds loaded from {path}", "error")

    def _cmd_load_recursive(self, args):
        """Force recursive directory walk, skipping the 4D sequence detector.

        Use this for hierarchical datasets where each leaf folder is a
        subject and you want every nested file registered as an
        independent sample. Example:

            load-recursive /path/to/dataset_root

        ``load`` auto-detects most nested trees, but this command is
        the explicit version for when auto-detection doesn't fire
        (e.g. the top level happens to contain a loose .ply).
        """
        if not args:
            self.log("Usage: load-recursive <path>", "error")
            return
        path = args[0]
        if not os.path.isdir(path):
            self.log(f"Directory not found: {path}", "error")
            return
        before = len(self.app.entries)
        self.app.load_directory_recursive(path)
        after = len(self.app.entries)
        loaded = after - before
        if loaded > 0:
            self.log(
                f"Registered {loaded} cloud(s) from {os.path.basename(path)}",
                "success",
            )
        else:
            self.log(f"No clouds registered from {path}", "error")

    def _cmd_list(self, args):
        if not self.app.entries:
            self.log("No clouds loaded.", "info")
            return
        for i, entry in enumerate(self.app.entries):
            marker = "*" if i == self.app.selected_index else " "
            pts = f"{entry.point_count:,}" if entry.point_count else "?"
            status = "ready" if entry.preview_gpu else "loading"
            self.log(f"  {marker} [{i}] {entry.name}  ({pts} pts, {status})", "info")

    def _cmd_select(self, args):
        if not args:
            self.log("Usage: select <index>", "error")
            return
        try:
            idx = int(args[0])
        except ValueError:
            self.log("Index must be a number.", "error")
            return
        if idx < 0 or idx >= len(self.app.entries):
            self.log(f"Index out of range (0-{len(self.app.entries)-1}).", "error")
            return
        self.app._on_cloud_selected(idx, enter_light_table=True)
        entry = self.app.entries[idx]
        self.log(f"Selected [{idx}] {entry.name}", "success")

    def _cmd_unload(self, args):
        if not args:
            self.log("Usage: unload <index|all>", "error")
            return
        if args[0].lower() == 'all':
            count = len(self.app.entries)
            for entry in self.app.entries:
                entry.release()
            self.app.entries.clear()
            self.app.gpu_clouds.clear()
            self.app.selected_index = 0
            self.app._overlays_dirty = True
            self.log(f"Unloaded all {count} cloud(s).", "success")
            return
        try:
            idx = int(args[0])
        except ValueError:
            self.log("Index must be a number or 'all'.", "error")
            return
        if idx < 0 or idx >= len(self.app.entries):
            self.log(f"Index out of range (0-{len(self.app.entries)-1}).", "error")
            return
        entry = self.app.entries.pop(idx)
        entry.release()
        if self.app.selected_index >= len(self.app.entries):
            self.app.selected_index = max(0, len(self.app.entries) - 1)
        self.app._overlays_dirty = True
        self.log(f"Unloaded [{idx}] {entry.name}", "success")

    def _cmd_info(self, args):
        if args:
            try:
                idx = int(args[0])
            except ValueError:
                self.log("Index must be a number.", "error")
                return
        else:
            idx = self.app.selected_index
        if idx < 0 or idx >= len(self.app.entries):
            self.log("No cloud at that index.", "error")
            return
        entry = self.app.entries[idx]
        self.log(f"Cloud [{idx}]: {entry.name}", "info")
        self.log(f"  Path: {entry.file_path}", "info")
        self.log(f"  Points: {entry.point_count:,}", "info")
        if entry.bounds_min is not None:
            bmin = entry.bounds_min
            bmax = entry.bounds_max
            self.log(f"  Bounds min: ({bmin[0]:.2f}, {bmin[1]:.2f}, {bmin[2]:.2f})", "info")
            self.log(f"  Bounds max: ({bmax[0]:.2f}, {bmax[1]:.2f}, {bmax[2]:.2f})", "info")
            size = bmax - bmin
            self.log(f"  Size: ({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})", "info")
        self.log(f"  Preview GPU: {'yes' if entry.preview_gpu else 'no'}", "info")
        self.log(f"  Full GPU: {'yes' if entry.full_gpu else 'no'}", "info")

    def _cmd_export(self, args):
        if not args:
            self.log("Usage: export <path> [--res WxH]", "error")
            return
        output = args[0]
        w, h = self._parse_res_args(args[1:])
        if not (0 <= self.app.selected_index < len(self.app.entries)):
            self.log("No cloud selected.", "error")
            return
        try:
            from src.export.image_export import export_still
            export_still(self.app, output, w, h)
            self.log(f"Exported {w}x{h} to {output}", "success")
        except Exception as e:
            self.log(f"Export failed: {e}", "error")

    def _cmd_spin(self, args):
        if not args:
            self.log("Usage: spin <dir> [--frames N] [--res WxH]", "error")
            return
        output_dir = args[0]
        w, h = self._parse_res_args(args[1:])
        frames = 360
        for i, a in enumerate(args[1:]):
            if a == '--frames' and i + 2 < len(args):
                try:
                    frames = int(args[i + 2])
                except ValueError:
                    pass
        if not (0 <= self.app.selected_index < len(self.app.entries)):
            self.log("No cloud selected.", "error")
            return
        try:
            from src.export.spin_export import export_spin
            self.log(f"Rendering {frames} frames at {w}x{h}...", "info")
            export_spin(self.app, output_dir, frames=frames, width=w, height=h)
            self.log(f"Spin render complete: {output_dir}", "success")
        except Exception as e:
            self.log(f"Spin render failed: {e}", "error")

    def _cmd_batch_export(self, args):
        if not args:
            self.log("Usage: batch-export <dir> [--res WxH]", "error")
            return
        output_dir = args[0]
        w, h = self._parse_res_args(args[1:])
        os.makedirs(output_dir, exist_ok=True)
        if not self.app.entries:
            self.log("No clouds loaded.", "error")
            return
        saved_idx = self.app.selected_index
        count = 0
        for i, entry in enumerate(self.app.entries):
            self.app.selected_index = i
            self.app._ensure_full_resolution(i)
            if entry.bounds_min is not None:
                self.app.camera.fit_to_bounds(entry.bounds_min, entry.bounds_max)
            name = os.path.splitext(entry.name)[0]
            out_path = os.path.join(output_dir, f"{name}.png")
            try:
                from src.export.image_export import export_still
                export_still(self.app, out_path, w, h)
                count += 1
                self.log(f"  [{i+1}/{len(self.app.entries)}] {entry.name} -> {out_path}", "info")
            except Exception as e:
                self.log(f"  [{i+1}/{len(self.app.entries)}] {entry.name} FAILED: {e}", "error")
        self.app.selected_index = saved_idx
        self.log(f"Batch export complete: {count}/{len(self.app.entries)} exported.", "success")

    def _cmd_export_dataset(self, args):
        """Export labeled clouds as a training dataset."""
        if not args:
            self.log("Usage: export-dataset <dir> [--format npz|ptv3|h5] [--split t,v,ts]", "error")
            return
        output_dir = args[0]
        fmt = 'ptv3'
        train, val, test = 0.7, 0.15, 0.15
        i = 1
        while i < len(args):
            if args[i] == '--format' and i + 1 < len(args):
                fmt = args[i + 1].lower()
                i += 2
            elif args[i] == '--split' and i + 1 < len(args):
                parts = args[i + 1].split(',')
                train, val, test = float(parts[0]), float(parts[1]), float(parts[2])
                i += 2
            else:
                i += 1

        if not self.app.entries:
            self.log("No clouds loaded.", "error")
            return

        # Collect full-res cloud data from each entry
        clouds = []
        for i, entry in enumerate(self.app.entries):
            self.app._ensure_full_resolution(i)
            gpu = entry.full_gpu or entry.preview_gpu
            if gpu is not None:
                clouds.append(gpu.cloud_data)
            else:
                self.log(f"Skipping {entry.name}: no data loaded", "error")

        if not clouds:
            self.log("No cloud data available to export.", "error")
            return

        try:
            from src.export.dataset_export import export_dataset
            summary = export_dataset(
                clouds, self.app.label_registry, output_dir, format=fmt,
                train_ratio=train, val_ratio=val, test_ratio=test,
            )
            self.log(f"Dataset exported: {summary['train']} train / {summary['val']} val / {summary['test']} test", "success")
            self.log(f"Format: {fmt}, Output: {output_dir}", "info")
        except Exception as e:
            self.log(f"Export failed: {e}", "error")

    def _cmd_label(self, args):
        """Create a label: label <name> [color_hex]"""
        if not args:
            self.log("Usage: label <name> [#rrggbb]", "error")
            return
        name = args[0]
        color = None
        if len(args) > 1 and args[1].startswith('#'):
            hex_str = args[1].lstrip('#')
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
            color = (r, g, b, 1.0)
        try:
            new_id = self.app.label_registry.add_label(name, color=color)
            if self.app.label_texture is not None:
                from src.rendering.label_texture import update_label_color_texture
                update_label_color_texture(self.app.label_texture, self.app.label_registry)
            self.log(f"Created label [{new_id}] '{name}'", "success")
        except Exception as e:
            self.log(f"Failed to create label: {e}", "error")

    def _cmd_labels(self, args):
        """List all labels with point counts on current cloud."""
        reg = self.app.label_registry
        if not reg.all_labels():
            self.log("No labels defined.", "info")
            return

        gpu = None
        if 0 <= self.app.selected_index < len(self.app.entries):
            entry = self.app.entries[self.app.selected_index]
            gpu = entry.full_gpu or entry.preview_gpu

        for info, depth in reg.get_hierarchy():
            prefix = "  " * depth
            count = 0
            if gpu is not None:
                import numpy as np
                count = int((gpu.cloud_data.labels == info.id).sum())
            marker = "*" if info.id == self.app.active_label_id else " "
            vis = "v" if info.visible else "-"
            lock = "L" if info.locked else " "
            self.log(f" {marker} [{info.id:3d}] {vis}{lock} {prefix}{info.name}  ({count:,} pts)", "info")

    def _cmd_train(self, args):
        """Launch a Pointcept PTv3 training subprocess.

        Usage: train <data_dir> [--epochs N] [--batch B] [--python-exe P] [--pointcept-dir D]
        """
        if not args:
            self.log("Usage: train <data_dir> [--epochs N] [--batch B]", "error")
            return
        if self.app.training_runner is not None and self.app.training_runner.is_running:
            self.log("Training is already running.", "error")
            return

        data_dir = args[0]
        epochs = int(getattr(self.app, '_train_epochs', 200))
        batch = int(getattr(self.app, '_train_batch', 2))
        python_exe = getattr(self.app, '_train_python_exe', '').strip()
        pointcept_dir = getattr(self.app, '_train_pointcept_dir', '').strip()

        i = 1
        while i < len(args):
            if args[i] == '--epochs' and i + 1 < len(args):
                epochs = int(args[i + 1]); i += 2
            elif args[i] == '--batch' and i + 1 < len(args):
                batch = int(args[i + 1]); i += 2
            elif args[i] == '--python-exe' and i + 1 < len(args):
                python_exe = args[i + 1]; i += 2
            elif args[i] == '--pointcept-dir' and i + 1 < len(args):
                pointcept_dir = args[i + 1]; i += 2
            else:
                i += 1

        if not python_exe or not pointcept_dir:
            self.log("Set python exe and pointcept dir first (TRAIN panel or --python-exe / --pointcept-dir).", "error")
            return

        from src.training.ptv3_runner import PointceptRunner, PointceptLaunchConfig
        from src.training.config_gen import PTv3TrainParams, generate_ptv3_config
        import time as _time

        num_classes = max(2, len(self.app.label_registry))
        run_name = f"run_{int(_time.time())}"
        work_dir = os.path.join(data_dir, "training_runs", run_name)
        os.makedirs(work_dir, exist_ok=True)

        params = PTv3TrainParams(
            data_root=data_dir, num_classes=num_classes,
            epochs=epochs, batch_size=batch,
        )
        config_path = os.path.join(work_dir, "config.py")
        ext_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "training", "pointcept_ext",
        )
        try:
            generate_ptv3_config(params, ext_dir, config_path)
        except Exception as e:
            self.log(f"Config generation failed: {e}", "error")
            return

        launch_cfg = PointceptLaunchConfig(
            python_exe=python_exe, pointcept_dir=pointcept_dir,
            config_file=config_path, work_dir=work_dir,
            pointcept_ext_dir=ext_dir,
        )
        if self.app.training_runner is None:
            self.app.training_runner = PointceptRunner()
        ok = self.app.training_runner.launch(launch_cfg)
        if ok:
            self.log(f"Pointcept PTv3 launched: {epochs} epochs, batch {batch}, {num_classes} classes", "success")
            self.log(f"  work_dir: {work_dir}", "info")
        else:
            self.log(f"Failed to launch: {self.app.training_runner.status.error}", "error")

    def _cmd_train_status(self, args):
        runner = self.app.training_runner
        if runner is None:
            self.log("No training runner initialized.", "info")
            return
        status = runner.status
        if status.running:
            step_info = ""
            if hasattr(status, 'current_step') and status.total_steps > 0:
                step_info = f" step {status.current_step}/{status.total_steps}"
            self.log(f"Training: epoch {status.current_epoch}/{status.total_epochs}{step_info}", "info")
            if status.last_loss > 0:
                self.log(f"  Loss: {status.last_loss:.4f}", "info")
            if status.last_miou > 0:
                self.log(f"  mIoU: {status.last_miou:.4f}", "info")
            if hasattr(status, 'best_miou') and status.best_miou > 0:
                self.log(f"  Best mIoU: {status.best_miou:.4f}", "info")
        elif status.finished:
            best = getattr(status, 'best_miou', 0)
            best_str = f"  best mIoU {best:.4f}" if best > 0 else ""
            self.log(f"Training finished at epoch {status.current_epoch}{best_str}", "success")
            if status.error:
                self.log(f"  Error: {status.error}", "error")
        else:
            self.log("Training not started.", "info")
        # Show last 5 log lines
        for line in status.log_lines[-5:]:
            self.log(f"  | {line}", "info")

    def _cmd_train_stop(self, args):
        runner = self.app.training_runner
        if runner is None or not runner.is_running:
            self.log("No training running.", "info")
            return
        runner.stop()
        self.log("Training stopped.", "success")

    def _cmd_save_project(self, args):
        if not args:
            self.log("Usage: save-project <path>", "error")
            return
        path = args[0]
        try:
            from src.data.project import AnnotationProject
            proj = AnnotationProject(self.app)
            proj.save(path)
            self.log(f"Project saved: {proj.project_path}", "success")
        except Exception as e:
            self.log(f"Save failed: {e}", "error")

    def _cmd_load_project(self, args):
        if not args:
            self.log("Usage: load-project <path>", "error")
            return
        path = args[0]
        try:
            from src.data.project import AnnotationProject, resolve_cloud_path
            # Clear current state
            for entry in self.app.entries:
                entry.release()
            self.app.entries.clear()
            self.app.gpu_clouds.clear()
            self.app.selected_index = 0

            proj = AnnotationProject(self.app)
            project_data = proj.load(path)
            # project.load replaces app.label_registry with a fresh
            # instance from registry.json; re-wire the schema auto-save
            # callback so subsequent label edits still persist.
            if hasattr(self.app, '_install_schema_callback'):
                self.app._install_schema_callback()

            # Load referenced cloud files
            for cloud_info in project_data.get('clouds', []):
                cloud_path = resolve_cloud_path(cloud_info, path)
                if cloud_path is not None:
                    self.app.load_file(cloud_path)
                else:
                    self.log(f"  Missing: {cloud_info.get('name')}", "error")

            # Apply saved labels
            proj.apply_labels_to_clouds(path)

            self.log(f"Loaded project: {len(self.app.entries)} clouds", "success")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log(f"Load failed: {e}", "error")

    def _cmd_propagate(self, args):
        """Propagate labels from current frame to next in the 4D sequence."""
        if self.app.sequence is None:
            self.log("No 4D sequence loaded.", "error")
            return
        seq = self.app.sequence
        if seq.current_index >= seq.frame_count - 1:
            self.log("Already at last frame.", "error")
            return

        # Parse args
        radius = None
        k = 5
        i = 0
        while i < len(args):
            if args[i] == '--radius' and i + 1 < len(args):
                radius = float(args[i + 1])
                i += 2
            elif args[i] == '--k' and i + 1 < len(args):
                k = int(args[i + 1])
                i += 2
            else:
                i += 1

        source_cloud = seq.current_frame()
        target_idx = seq.current_index + 1
        target_cloud = seq.get_frame(target_idx)

        # Refuse to propagate from an unlabeled source — doing so would
        # overwrite the target with all-zero labels silently.
        if source_cloud.labels is None or not (source_cloud.labels != 0).any():
            self.log(
                f"Source frame {seq.current_index + 1} has no labeled points — nothing to propagate.",
                "error",
            )
            return

        from src.core.propagation import propagate_labels, estimate_radius
        if radius is None:
            radius = estimate_radius(source_cloud.positions)
            self.log(f"Auto radius: {radius:.4f}", "info")

        try:
            propagated, confidences = propagate_labels(
                source_cloud.positions, source_cloud.labels,
                target_cloud.positions, radius=radius, k=k,
            )
        except Exception as e:
            self.log(f"Propagation failed: {e}", "error")
            return

        # LS-8: route the in-memory mutation through apply_label so
        # propagation becomes undoable. We loop over distinct
        # propagated label ids — one LabelCommand per destination
        # label. The user can Ctrl+Z N times to back out a single
        # propagate.
        from src.core.undo import apply_label
        unique_ids = np.unique(propagated)
        for lid in unique_ids:
            idx = np.where(propagated == lid)[0].astype(np.int32)
            if idx.size == 0:
                continue
            apply_label(target_cloud, idx, int(lid),
                        self.app.undo_stack,
                        description=f"propagate frame -> label {lid}")
        target_cloud.scalars['propagation_confidence'] = confidences

        labeled_count = int((propagated != 0).sum())
        mean_conf = float(confidences[confidences > 0].mean()) if (confidences > 0).any() else 0.0
        self.log(f"Propagated to frame {target_idx + 1}: {labeled_count:,} pts, mean conf {mean_conf:.2f}", "success")

        # Auto-advance to the new frame
        self.app._seek_sequence(target_idx)

    def _cmd_propagate_all(self, args):
        """Propagate labels through all remaining frames from current."""
        if self.app.sequence is None:
            self.log("No 4D sequence loaded.", "error")
            return
        seq = self.app.sequence

        radius = None
        i = 0
        while i < len(args):
            if args[i] == '--radius' and i + 1 < len(args):
                radius = float(args[i + 1])
                i += 2
            else:
                i += 1

        start = seq.current_index
        from src.core.propagation import propagate_labels, estimate_radius
        count = 0
        for source_idx in range(start, seq.frame_count - 1):
            source_cloud = seq.get_frame(source_idx)
            target_cloud = seq.get_frame(source_idx + 1)
            # Skip frames whose source has nothing to propagate.
            if source_cloud.labels is None or not (source_cloud.labels != 0).any():
                self.log(
                    f"  Frame {source_idx + 1}: no labels — skipping propagation.",
                    "info",
                )
                continue
            r = radius if radius is not None else estimate_radius(source_cloud.positions)
            propagated, confidences = propagate_labels(
                source_cloud.positions, source_cloud.labels,
                target_cloud.positions, radius=r, k=5,
            )
            # LS-8: route through apply_label so propagate-all stays
            # undoable, same as single-frame propagate above.
            from src.core.undo import apply_label
            for lid in np.unique(propagated):
                idx = np.where(propagated == lid)[0].astype(np.int32)
                if idx.size == 0:
                    continue
                apply_label(target_cloud, idx, int(lid),
                            self.app.undo_stack,
                            description=f"propagate-all -> label {lid}")
            target_cloud.scalars['propagation_confidence'] = confidences
            count += 1
            labeled = int((propagated != 0).sum())
            self.log(f"  Frame {source_idx + 2}: {labeled:,} pts", "info")

        self.log(f"Propagated through {count} frames.", "success")

    def _cmd_set(self, args):
        if len(args) < 2:
            self.log("Usage: set <key> <value>", "error")
            self.log("  Keys: point_size, sharpness, brightness, contrast, saturation, bg, bbox, grid", "info")
            return
        key = args[0].lower()
        val = args[1]
        if key == 'point_size':
            self.app.point_size = float(val)
            self.log(f"point_size = {self.app.point_size:.1f}", "success")
        elif key == 'sharpness':
            self.app.point_sharpness = float(val)
            self.log(f"sharpness = {self.app.point_sharpness:.2f}", "success")
        elif key == 'brightness':
            self.app.brightness = float(val)
            self.log(f"brightness = {self.app.brightness:.2f}", "success")
        elif key == 'contrast':
            self.app.contrast = float(val)
            self.log(f"contrast = {self.app.contrast:.2f}", "success")
        elif key == 'saturation':
            self.app.saturation = float(val)
            self.log(f"saturation = {self.app.saturation:.2f}", "success")
        elif key == 'bg':
            parts = val.split(',')
            self.app.bg_color = tuple(float(x) for x in parts[:3])
            self.log(f"bg = {self.app.bg_color}", "success")
        elif key == 'bbox':
            self.app.show_bbox = val.lower() in ('true', '1', 'on', 'yes')
            self.log(f"bbox = {self.app.show_bbox}", "success")
        elif key == 'grid':
            self.app.show_grid = val.lower() in ('true', '1', 'on', 'yes')
            self.log(f"grid = {self.app.show_grid}", "success")
        else:
            self.log(f"Unknown key: '{key}'", "error")

    def _cmd_get(self, args):
        if not args:
            self.log("Usage: get <key>", "error")
            return
        key = args[0].lower()
        vals = {
            'point_size': f"{self.app.point_size:.1f}",
            'sharpness': f"{self.app.point_sharpness:.2f}",
            'brightness': f"{self.app.brightness:.2f}",
            'contrast': f"{self.app.contrast:.2f}",
            'saturation': f"{self.app.saturation:.2f}",
            'bg': f"{self.app.bg_color}",
            'bbox': f"{self.app.show_bbox}",
            'grid': f"{self.app.show_grid}",
        }
        if key in vals:
            self.log(f"{key} = {vals[key]}", "info")
        else:
            self.log(f"Unknown key: '{key}'. Keys: {', '.join(vals)}", "error")

    def _cmd_camera(self, args):
        if not args:
            self.log("Usage: camera <preset>", "error")
            self.log("  Presets: top, front, right, iso, fit", "info")
            return
        preset = args[0].lower()
        if preset == 'fit':
            self.app._fit_camera()
            self.log("Camera fit to bounds.", "success")
        else:
            try:
                self.app.camera.set_preset(preset)
                self.log(f"Camera preset: {preset}", "success")
            except Exception:
                self.log(f"Unknown preset: '{preset}'", "error")

    def _cmd_clear(self, args):
        self.output.clear()

    def _cmd_status(self, args):
        mode_names = {MODE_CONTACT_SHEETS: "Contact Sheets", MODE_LIGHT_TABLE: "Light Table", MODE_AUTOMATION: "Train"}
        mode = mode_names.get(self.app.mode, "Unknown")
        self.log(f"Mode: {mode}", "info")
        self.log(f"Clouds: {len(self.app.entries)}", "info")
        total = sum(e.point_count for e in self.app.entries)
        self.log(f"Total points: {total:,}", "info")
        self.log(f"Selected: [{self.app.selected_index}]", "info")
        self.log(f"Point size: {self.app.point_size:.1f}", "info")
        self.log(f"Sharpness: {self.app.point_sharpness:.2f}", "info")
        self.log(f"Brightness: {self.app.brightness:.2f}  Contrast: {self.app.contrast:.2f}  Saturation: {self.app.saturation:.2f}", "info")
        self.log(f"BBox: {self.app.show_bbox}  Grid: {self.app.show_grid}", "info")
        pending = self.app.catalog.pending_count if self.app.catalog else 0
        if pending:
            self.log(f"Pending loads: {pending}", "info")

    def _cmd_catalog_status(self, args):
        """Read-only catalog integrity report (entries, orphans, missing)."""
        from src.data.cloud_store import (
            check_catalog_integrity, format_integrity_summary, estimate_catalog_size
        )
        try:
            status = check_catalog_integrity(catalog=self.app.catalog)
        except Exception as e:
            self.log(f"Catalog check failed: {e}", "error")
            return
        self.log(format_integrity_summary(status), "info")
        size_bytes = estimate_catalog_size()
        if size_bytes > 0:
            mb = size_bytes / (1024.0 * 1024.0)
            self.log(f"Disk usage: {mb:,.1f} MB", "info")
        if status.get("orphans"):
            self.log(
                f"  Orphan files (data/labels with no index entry): "
                f"{len(status['orphans'])}",
                "info",
            )
            for k in status["orphans"][:5]:
                self.log(f"    {k}", "info")
            if len(status["orphans"]) > 5:
                self.log(f"    ... and {len(status['orphans']) - 5} more", "info")
        if status.get("missing_data"):
            self.log(
                f"  Index entries with no data file: {len(status['missing_data'])}",
                "info",
            )

    def _cmd_wipe_catalog(self, args):
        """Wipe the entire library catalog. Requires --confirm to act.

        Without --confirm: print a summary of what would be lost so the
        user can review before committing. With --confirm: tar.gz the
        whole catalog to ~/.lithium/backups/library_<ts>.tar.gz, then
        clear data/, labels/, previews/, index.json, schema.json,
        projects.json, and the .lock file. The empty subdirectory
        skeleton is left in place. App state (loaded clouds, registry)
        is reset so the running session reflects the wipe.
        """
        from src.data.cloud_store import (
            check_catalog_integrity, format_integrity_summary,
            wipe_catalog, estimate_catalog_size,
        )
        from src.data.labels import LabelRegistry

        confirm = '--confirm' in args
        no_backup = '--no-backup' in args

        try:
            status = check_catalog_integrity(catalog=self.app.catalog)
        except Exception as e:
            self.log(f"Catalog check failed: {e}", "error")
            return

        if not confirm:
            self.log("wipe-catalog will erase the entire Lithium library:", "info")
            self.log(f"  {format_integrity_summary(status)}", "info")
            size_bytes = estimate_catalog_size()
            if size_bytes > 0:
                mb = size_bytes / (1024.0 * 1024.0)
                self.log(f"  Disk usage to back up: {mb:,.1f} MB", "info")
            self.log("Re-run as `wipe-catalog --confirm` to proceed.", "info")
            self.log("Add `--no-backup` to skip the tarball (DESTRUCTIVE).", "info")
            return

        try:
            backup_path = wipe_catalog(backup=not no_backup)
        except Exception as e:
            self.log(f"Wipe failed: {e}", "error")
            return

        if backup_path is not None:
            self.log(f"Backup written to {backup_path}", "success")
        elif not no_backup:
            self.log("Wipe aborted: backup step failed (catalog untouched).", "error")
            return

        # Reset live app state so the running session reflects the wipe.
        try:
            for entry in self.app.entries:
                entry.release()
        except Exception:
            pass
        self.app.entries.clear()
        if hasattr(self.app, 'gpu_clouds'):
            self.app.gpu_clouds.clear()
        self.app.selected_index = 0

        # Fresh empty registry. Labels are project-scoped now, so a
        # wipe just starts us at zero with no active project.
        self.app.label_registry = LabelRegistry()
        self.app.label_registry._on_change_callback = None
        if hasattr(self.app, 'label_count_cache'):
            self.app.label_count_cache.clear()

        # Recreate an empty live catalog so the gallery doesn't crash.
        try:
            from src.data.library_catalog import LibraryCatalog
            if self.app.catalog is not None:
                self.app.catalog.shutdown()
            self.app.catalog = LibraryCatalog()
        except Exception as e:
            self.log(f"Catalog reinit failed: {e}", "error")

        self.log("Catalog wiped. 0 entries, 0 labels, 0 schema.", "success")

    def _parse_res_args(self, args) -> tuple[int, int]:
        for i, a in enumerate(args):
            if a == '--res' and i + 1 < len(args):
                try:
                    w, h = args[i + 1].split('x')
                    return int(w), int(h)
                except (ValueError, IndexError):
                    pass
        return 1920, 1080
