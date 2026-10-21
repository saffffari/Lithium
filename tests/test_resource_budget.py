"""Cache bounds must preserve active geometry and never discard failed saves."""
from concurrent.futures import Future
from types import SimpleNamespace
import numpy as np
import pytest
from src.core.resource_budget import thumbnail_side, eviction_candidates, cloud_payload_bytes
from src.data.point_cloud import PointCloudData


def gpu(n=10):
    cloud = PointCloudData(positions=np.zeros((n,3),np.float32), colors=np.zeros((n,3),np.float32))
    return SimpleNamespace(point_count=n, cloud_data=cloud, _sel_staging=None,
                           _lbl_staging=cloud.labels, release=lambda: None)


def entry(i):
    return SimpleNamespace(full_gpu=gpu(), preview_gpu=None, mesh_gpu=None,
        file_key=str(i), _resident_touch=i, _preview_to_full_idx_cache=None)


def test_thumbnail_budget_scales_texture_not_layout():
    side = thumbnail_side(1024, 120, 8 * 120 * 128**2)
    assert side == 128
    assert side**2 * 8 * 120 <= 8 * 120 * 128**2
    assert thumbnail_side(64, 10) == 64


def test_payload_counts_label_staging_alias_once():
    g = gpu(10)
    assert cloud_payload_bytes(g) == 10 * (32 + 12 + 12 + 4)


def test_cold_lru_is_evicted_and_active_is_pinned_even_over_budget():
    entries = [entry(i) for i in range(6)]
    victims = list(eviction_candidates(entries, 'full_gpu', {id(entries[0])}, 3, 10**9))
    assert victims == entries[1:4]
    victims = list(eviction_candidates(entries, 'full_gpu', {id(entries[0])}, 1, 1))
    assert victims == entries[1:]
    assert entries[0].full_gpu.cloud_data.point_count == 10


def test_app_trims_verified_copies_but_keeps_failed_save_and_active(monkeypatch):
    from src.main import App
    app = App()
    try:
        app.entries = [entry(i) for i in range(6)]
        app.selected_index = 0
        failed = app.entries[1]
        app._persist_cloud_labels = lambda e,c: e is not failed
        released = []
        for i,e in enumerate(app.entries):
            e.full_gpu.release = lambda i=i: released.append(i)
        app.gpu_clouds = [e.full_gpu for e in app.entries]
        app._trim_resident_clouds(force=True)
        assert app.entries[0].full_gpu is not None
        assert failed.full_gpu is not None
        assert released and all(i not in (0,1) for i in released)
        assert len(app.gpu_clouds) == 6 - len(released)
    finally:
        app._full_res_executor.shutdown(wait=True)
        app._preview_executor.shutdown(wait=True)


def test_background_load_from_old_project_cannot_land_on_same_cloud(monkeypatch):
    from src.main import App
    from src.data import cloud_store
    app = App()
    original_ns = cloud_store.active_label_namespace()
    try:
        app.entries = [entry(0)]
        app.entries[0].full_gpu = None
        future = Future()
        future.set_result((gpu().cloud_data, False, None))
        app._full_res_pending = (0, future, np.eye(4), '0', 'project-a')
        cloud_store.set_active_label_namespace('project-b')
        app._poll_full_res()
        assert app.entries[0].full_gpu is None
        assert app._full_res_pending is None
    finally:
        cloud_store.set_active_label_namespace(original_ns)
        app._full_res_executor.shutdown(wait=True)
        app._preview_executor.shutdown(wait=True)


def test_gallery_cache_lru_obeys_bytes_instead_of_only_count():
    from collections import OrderedDict
    from src.rendering.gallery_cache import GalleryCache
    cache = GalleryCache(None, max_bytes=8 * 256**2 * 2)
    cache._cell_size = 256
    released = []
    cache._cells = OrderedDict((i, SimpleNamespace(size=256, release=lambda i=i: released.append(i))) for i in range(5))
    cache._evict_if_needed()
    assert list(cache._cells) == [3,4]
    assert released == [0,1,2]
    assert cache.texture_bytes <= cache.max_bytes


@pytest.mark.parametrize('ram,vram,renderer,expected', [
    (8,0,'Intel UHD','Lean'), (16,0,'unknown','Lean'),
    (16,4,'NVIDIA GTX','Balanced'), (128,24,'NVIDIA RTX 4090','Workstation'),
    (128,24,'llvmpipe','Lean'),
])
def test_auto_scales_to_hardware_and_keeps_workstation_resources(ram,vram,renderer,expected):
    from src.core.resource_budget import choose_budget
    assert choose_budget('auto',ram*2**30,vram*2**30,renderer).name == expected
    assert choose_budget('workstation',8*2**30,0,'unknown').name == 'Workstation'
    assert choose_budget('lean',128*2**30,24*2**30,'RTX').name == 'Lean'


def test_preview_queue_is_bounded_and_captures_namespace(tmp_path):
    from src.main import App, CloudEntry
    from src.data import cloud_store
    class PendingExecutor:
        def __init__(self): self.calls=[]
        def submit(self,fn,*args):
            self.calls.append(args)
            return Future()
        def shutdown(self,**kwargs): pass
    app=App();old_ns=cloud_store.active_label_namespace()
    app._preview_executor.shutdown(wait=True)
    app._preview_executor=PendingExecutor()
    try:
        cloud_store.set_active_label_namespace('preview-project-a')
        app.catalog=SimpleNamespace(dir=str(tmp_path),entries={})
        for i in range(12):
            path=tmp_path/f'{i}.npz';path.touch()
            app.catalog.entries[str(i)]=SimpleNamespace(preview_path=lambda d,path=path:str(path))
            assert app._ensure_preview_gpu(CloudEntry(str(path),file_key=str(i))) is False
        assert len(app._preview_pending)==4
        assert len(app._preview_executor.calls)==4
        assert all(call[2]=='preview-project-a' for call in app._preview_executor.calls)
    finally:
        cloud_store.set_active_label_namespace(old_ns)
        app._full_res_executor.shutdown(wait=True)


def test_preview_worker_uses_captured_project_labels(tmp_path,monkeypatch):
    from src.main import App
    from src.data import cloud_store
    monkeypatch.setenv('LITHIUM_LIBRARY_DIR',str(tmp_path/'library'))
    path=tmp_path/'preview.npz'
    np.savez(path,positions=np.zeros((10,3),np.float32),colors=np.ones((10,3),np.float32))
    cloud_store.save_preview_labels('fk',np.full(10,2,np.int32),namespace='A')
    cloud_store.save_preview_labels('fk',np.full(10,3,np.int32),namespace='B')
    app=App();original=cloud_store.active_label_namespace()
    try:
        cloud_store.set_active_label_namespace('B')
        preview,_=app._preview_load_job('fk',str(path),'A')
        np.testing.assert_array_equal(preview.labels,np.full(10,2))
    finally:
        cloud_store.set_active_label_namespace(original)
        app._full_res_executor.shutdown(wait=True)
        app._preview_executor.shutdown(wait=True)


def test_failed_mesh_build_is_not_requeued_every_frame(tmp_path,monkeypatch):
    from src.data.library_catalog import LibraryCatalog,LibraryEntry
    from src.data import cloud_store
    monkeypatch.setenv('LITHIUM_LIBRARY_DIR',str(tmp_path))
    cat=LibraryCatalog()
    try:
        e=LibraryEntry(file_key='mesh-test',file_path='missing.npz')
        failed=Future();failed.set_result(('mesh-test','optional mesh backend missing'))
        cat._pending_mesh_futures=[(e.file_key,failed)]
        assert cat.poll_pending_meshes()==[]
        assert cat.mesh_errors[e.file_key]=='optional mesh backend missing'
        cat.queue_mesh_build(e)
        assert not cat._pending_mesh_futures
    finally:
        cat.shutdown()
