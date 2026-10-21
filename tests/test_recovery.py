"""Interrupted-stroke recovery must preserve canonical and project isolation."""
import os
import numpy as np
import pytest
from src.data import cloud_store
from src.data.point_cloud import PointCloudData


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    monkeypatch.setenv('LITHIUM_LIBRARY_DIR', str(tmp_path))
    fk = '0123456789abcdef'
    cloud_store.save_cloud_data(fk, PointCloudData(
        positions=np.zeros((10, 3), np.float32), colors=np.ones((10, 3), np.float32)))
    saved = np.zeros(10, np.int32)
    cloud_store.save_cloud_labels(fk, saved, namespace='project-a')
    path = cloud_store.cloud_labels_path(fk, 'project-a')
    sidecar = path.with_name(fk + '.recovery.npy')
    np.save(sidecar, np.ones(10, np.int32))
    now = path.stat().st_mtime_ns + 1000000000
    os.utime(sidecar, ns=(now, now))
    return fk, path, sidecar


def test_restore_backs_up_commit_and_cleans_snapshot(recovery):
    fk, path, sidecar = recovery
    np.testing.assert_array_equal(cloud_store.load_cloud_labels(fk, 'project-a'), np.zeros(10))
    assert cloud_store.pending_label_recovery(fk, 'project-b') is None
    assert cloud_store.resolve_label_recovery(fk, True, 'project-a') is None
    np.testing.assert_array_equal(np.load(path), np.ones(10))
    assert not sidecar.exists()
    backups = list(path.parents[2].glob('backups/**/*.npy'))
    assert any(np.array_equal(np.load(p), np.zeros(10)) for p in backups)


def test_keep_commit_discards_only_snapshot(recovery):
    fk, path, sidecar = recovery
    assert cloud_store.resolve_label_recovery(fk, False, 'project-a') is None
    np.testing.assert_array_equal(np.load(path), np.zeros(10))
    assert not sidecar.exists()


@pytest.mark.parametrize('array', [np.ones(9, np.int32), np.ones((10,1), np.int32), np.ones(10, np.float32)])
def test_invalid_snapshot_cannot_replace_saved_labels(recovery, array):
    fk, path, sidecar = recovery
    np.save(sidecar, array)
    now = path.stat().st_mtime_ns + 1000000000
    os.utime(sidecar, ns=(now, now))
    assert cloud_store.pending_label_recovery(fk, 'project-a') is None
    assert cloud_store.resolve_label_recovery(fk, True, 'project-a')
    np.testing.assert_array_equal(np.load(path), np.zeros(10))
    assert sidecar.exists()


def test_verified_save_removes_old_snapshot(recovery):
    fk, path, sidecar = recovery
    assert cloud_store.save_cloud_labels(fk, np.full(10, 2), namespace='project-a') is None
    assert not sidecar.exists()
    np.testing.assert_array_equal(np.load(path), np.full(10, 2))


def test_restore_refuses_to_overwrite_when_backup_fails(recovery,monkeypatch):
    fk,path,sidecar=recovery
    monkeypatch.setattr(cloud_store,'snapshot_labels',lambda *a:(path.parent,0,1))
    assert cloud_store.resolve_label_recovery(fk,True,'project-a')
    np.testing.assert_array_equal(np.load(path),np.zeros(10))
    assert sidecar.exists()
