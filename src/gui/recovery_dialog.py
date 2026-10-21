"""Offer interrupted-stroke recovery before the autosave timer can overwrite it."""
import imgui
from src.data import cloud_store


def draw_recovery_dialog(app):
    checked = getattr(app, '_recovery_checked', None)
    if checked is None:
        checked = app._recovery_checked = set()
    pending = getattr(app, '_pending_recovery', None)
    if pending is None:
        for entry in app.entries:
            key = (str(getattr(app, 'active_view', None)), entry.file_key)
            if key in checked:
                continue
            if cloud_store.pending_label_recovery(entry.file_key) is not None:
                pending = app._pending_recovery = (key, entry, cloud_store.active_label_namespace())
                break
            checked.add(key)
    if pending is None:
        return
    key, entry, namespace = pending
    imgui.open_popup('Recover interrupted stroke?')
    opened, _ = imgui.begin_popup_modal('Recover interrupted stroke?',
        flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)
    if not opened:
        return
    imgui.text('An unfinished stroke was saved before the last interruption.')
    imgui.text(entry.name)
    imgui.text('Restore keeps a backup of your last committed labels.')
    error = getattr(app, '_recovery_error', '')
    if error:
        imgui.text_wrapped(error)
    action = None
    if imgui.button('Restore snapshot'):
        action = True
    imgui.same_line()
    if imgui.button('Keep last committed labels'):
        action = False
    if action is not None:
        error = cloud_store.resolve_label_recovery(entry.file_key, action, namespace=namespace, catalog=app.catalog)
        app._recovery_error = error or ''
        if not error:
            checked.add(key)
            app._pending_recovery = None
            labels = cloud_store.load_cloud_labels(entry.file_key, namespace)
            gpu = entry.full_gpu
            same_namespace = (cloud_store.active_label_namespace() == namespace
                              and any(e is entry for e in app.entries))
            if same_namespace and gpu is not None and labels is not None and len(labels) == gpu.cloud_data.point_count:
                gpu.cloud_data.labels[:] = labels
                from src.rendering.point_cloud_renderer import re_upload_labels
                gpu.cloud_data.labels_version += 1
                re_upload_labels(gpu)
                if entry.preview_gpu is not None and entry.preview_gpu is not gpu:
                    app._propagate_labels_to_preview(entry)
                entry.mesh_dirty = True
            elif same_namespace and entry.preview_gpu is not None:
                gpu = entry.preview_gpu
                labels = app._derive_preview_labels_from_full(entry.file_key, gpu.cloud_data)
                if labels is not None:
                    from src.rendering.point_cloud_renderer import re_upload_labels
                    gpu.cloud_data.labels[:] = labels
                    gpu.cloud_data.labels_version += 1
                    re_upload_labels(gpu)
                    cloud_store.save_preview_labels(entry.file_key, labels, namespace=namespace)
            app.label_count_cache.clear()
            imgui.close_current_popup()
    imgui.end_popup()
