"""T13 — migration upgrade, rollback and encrypted restore succeed on seeded test data, with held
records preserved (S1.12 gate; OPS-01).

The seeded world: the T01 matters, an indexed document in M-101 and one in M-202, M-202 under an
approved legal hold, and the audit trail all of that produced. Restores go into a fresh folder and
the app is pointed at it; nothing is mocked except the embedding model.
"""
from __future__ import annotations

import io
import os
import sqlite3
import tarfile
from contextlib import closing

import pytest

from conftest import as_user, csrf_headers, index_all, make_user, search_in, upload

MEMO = "Settlement memo (fictional). The KESTREL-5150 figure is privileged and confidential."
HELD = "Board minutes (fictional). The PETREL-2020 resolution approved the investigation budget."


@pytest.fixture
def seeded(docs):
    for name in ("rhea", "rex"):
        docs["users"][name] = make_user(name, mfa=True, firm_roles=["records_admin"])
    alice, bob = as_user(docs, "alice"), as_user(docs, "bob")
    m101, m202 = docs["matters"]["M-101"]["id"], docs["matters"]["M-202"]["id"]
    memo = upload(alice, m101, "settlement-memo.txt", MEMO).json()["document"]
    held = upload(bob, m202, "board-minutes.txt", HELD).json()["document"]
    index_all()
    rhea, rex = as_user(docs, "rhea"), as_user(docs, "rex")
    hold = rhea.post("/api/legal/holds", headers=csrf_headers(rhea),
                     json={"name": "Beta investigation hold", "reason": "Anticipated claim", "matter_ids": [m202]}).json()["hold"]
    assert rex.post(f"/api/legal/holds/{hold['id']}/approve", headers=csrf_headers(rex)).status_code == 200
    docs.update(m101=m101, m202=m202, memo=memo, held=held, hold=hold)
    return docs


@pytest.fixture
def backup_key(tmp_path):
    from app.legal import backup
    os.makedirs(tmp_path / "keys", mode=0o700)
    return backup.generate_key(str(tmp_path / "keys" / "backup.key"))


def use_data(monkeypatch, folder: str) -> None:
    from app.core.config import settings
    from app.legal.retrieval import registry
    monkeypatch.setattr(settings, "DATA_DIR", folder)
    monkeypatch.setattr(settings, "DB_PATH", os.path.join(folder, "echomind.sqlite"))
    registry.clear()                                         # partitions are rebuilt from the restored database


def held_deletion(c, w):
    return c.post(f"/api/legal/matters/{w['m202']}/documents/{w['held']['id']}/delete", headers=csrf_headers(c),
                  json={"reason": "Duplicate"})


# ── Encrypted backup and restore ─────────────────────────────────────────────────
def test_t13_an_encrypted_backup_restores_everything_with_the_hold_in_force(seeded, backup_key, tmp_path, monkeypatch):
    from app.legal import backup
    from app.legal.audit import chain
    from app.legal.migrations import runner
    alice, bob = as_user(seeded, "alice"), as_user(seeded, "bob")
    file_url = (f"/api/legal/matters/{seeded['m101']}/documents/{seeded['memo']['id']}/versions/"
                f"{seeded['memo']['current_version']['id']}/file")
    first = alice.get(file_url)
    assert first.status_code == 200 and MEMO.encode() in first.content
    original = first.content
    made = backup.create(str(tmp_path / "backups" / "law.emlb"), backup_key)
    assert made["files"] >= 3 and made["counts"]["active_holds"] == 1 and made["schema"] == list(range(1, 13))

    sealed = open(made["archive"], "rb").read()
    for plain in (b"KESTREL-5150", b"PETREL-2020", b"Beta Corp", b"M-202", b"alice", b"SQLite format 3"):
        assert plain not in sealed, plain                    # nothing readable without the backup key
    assert os.stat(made["archive"]).st_mode & 0o077 == 0

    restored = backup.restore(made["archive"], backup_key, str(tmp_path / "restored"))
    assert restored["counts"] == made["counts"] and restored["audit"] == made["audit"]
    use_data(monkeypatch, restored["target"])

    assert chain.verify()["ok"] is True                      # chain and anchors came back together
    with closing(sqlite3.connect(restored["database"])) as c:
        assert all(r["state"] == "applied" for r in runner.status(c))
    got = alice.get(file_url)
    assert got.status_code == 200 and got.content == original                    # opens with the same master key
    hits = search_in(alice, seeded["m101"], "What is the KESTREL-5150 figure?").json()["passages"]
    assert hits and "KESTREL-5150" in hits[0]["text"]
    r = held_deletion(bob, seeded)
    assert r.status_code == 409 and "legal hold" in r.json()["detail"]            # T04 still holds after restore
    assert search_in(alice, seeded["m202"], "PETREL-2020").status_code == 404     # walls and membership restored too


def test_t13_a_backup_is_recorded_in_the_audit_trail(seeded, backup_key, tmp_path, fresh_db):
    from app.legal import backup
    backup.create(str(tmp_path / "b" / "law.emlb"), backup_key)
    with closing(sqlite3.connect(fresh_db)) as c:
        row = c.execute("SELECT actor_username, object_type, details_json FROM audit_events WHERE action='backup.create'").fetchone()
    assert row[0] == "operator:cli" and row[1] == "backup" and '"files"' in row[2]


def test_t13_restore_refuses_a_wrong_key_an_altered_or_cut_archive_and_a_used_folder(seeded, backup_key, tmp_path):
    from app.legal import backup
    made = backup.create(str(tmp_path / "b" / "law.emlb"), backup_key)
    data = open(made["archive"], "rb").read()

    other = backup.generate_key(str(tmp_path / "keys" / "other.key"))
    with pytest.raises(backup.BackupError, match="different backup key"):
        backup.restore(made["archive"], other, str(tmp_path / "r1"))

    def variant(name: str, content: bytes) -> str:
        path = tmp_path / "b" / name
        path.write_bytes(content)
        return str(path)
    flipped = bytearray(data)
    flipped[len(data) // 2] ^= 0x01
    for archive, why in ((variant("flipped.emlb", bytes(flipped)), "does not authenticate"),
                         (variant("cut.emlb", data[: len(data) - 40]), "truncated|does not authenticate"),
                         (variant("cut-at-chunk.emlb", data[: len(data) // 2]), "truncated|does not authenticate"),
                         (variant("tail.emlb", data + b"x"), "after the end|truncated"),
                         (variant("other.emlb", b"PK\x03\x04 not a backup"), "not an EchoMind Law backup")):
        with pytest.raises(backup.BackupError, match=why):
            backup.restore(archive, backup_key, str(tmp_path / f"r-{os.path.basename(archive)}"))
        assert not os.listdir(tmp_path / f"r-{os.path.basename(archive)}")        # nothing written before it authenticated

    used = tmp_path / "used"
    used.mkdir()
    (used / "keep.txt").write_text("existing data")
    with pytest.raises(backup.BackupError, match="new or empty folder"):
        backup.restore(made["archive"], backup_key, str(used))
    assert os.listdir(used) == ["keep.txt"]


def _sealed_tar(path: str, key_file: str, entries) -> str:
    from app.legal import backup
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for info, body in entries:
            tar.addfile(info, io.BytesIO(body) if body is not None else None)
    with open(path, "wb") as raw:
        w = backup._SealingWriter(raw, backup.read_key(key_file))
        w.write(buf.getvalue())
        w.finish()
    return path


@pytest.mark.parametrize("make", ["traversal", "absolute", "symlink", "device"])
def test_t13_restore_refuses_entries_outside_the_data_folder(backup_key, tmp_path, make):
    from app.legal import backup
    info = tarfile.TarInfo({"traversal": "data/../../escaped.txt", "absolute": "/tmp/escaped.txt",
                            "symlink": "data/link", "device": "data/dev"}[make])
    body = b"x"
    if make == "symlink":
        info.type, info.linkname, body = tarfile.SYMTYPE, "/etc/passwd", None
    elif make == "device":
        info.type, body = tarfile.CHRTYPE, None
    else:
        info.size = 1
    archive = _sealed_tar(str(tmp_path / "evil.emlb"), backup_key, [(info, body)])
    with pytest.raises(backup.BackupError, match="not allowed|outside the data folder"):
        backup.restore(archive, backup_key, str(tmp_path / "restore" / "data"))
    assert not (tmp_path / "escaped.txt").exists() and not (tmp_path / "restore" / "escaped.txt").exists()


def test_t13_keys_are_private_and_never_overwritten_and_backups_stay_out_of_the_data_folder(seeded, tmp_path, backup_key):
    from app.core.config import settings
    from app.legal import backup
    loose = tmp_path / "keys" / "loose.key"
    loose.write_text(open(backup_key).read())
    loose.chmod(0o644)
    with pytest.raises(backup.BackupError, match="only by its owner"):
        backup.create(str(tmp_path / "b" / "x.emlb"), str(loose))
    with pytest.raises(backup.BackupError, match="never overwritten"):
        backup.generate_key(backup_key)
    with pytest.raises(backup.BackupError, match="outside the data folder"):
        backup.create(os.path.join(settings.DATA_DIR, "inside.emlb"), backup_key)


def test_t13_the_cli_round_trip(seeded, tmp_path, capsys):
    from app.legal.backup.__main__ import main
    key = str(tmp_path / "cli.key")
    assert main(["keygen", "--out", key]) == 0
    archive = str(tmp_path / "cli" / "law.emlb")
    assert main(["create", "--out", archive, "--key-file", key]) == 0
    assert main(["verify", "--archive", archive, "--key-file", key]) == 0
    assert main(["restore", "--archive", archive, "--key-file", key, "--target", str(tmp_path / "cli-restored")]) == 0
    assert main(["restore", "--archive", archive, "--key-file", key, "--target", str(tmp_path / "cli-restored")]) == 1
    assert "new or empty folder" in capsys.readouterr().err


# ── Upgrade and rollback ─────────────────────────────────────────────────────────
def test_t13_upgrading_from_the_s1_0_schema_keeps_legacy_data(fresh_db, monkeypatch):
    from app.core import db
    from app.legal.migrations import runner
    monkeypatch.delenv("ECHOMIND_EDITION", raising=False)
    db.init_db()                                             # the standard edition's tables
    with closing(sqlite3.connect(fresh_db)) as c:
        runner.migrate_up(c, target=1)                       # what S1.0 shipped
        c.execute("INSERT INTO users(id, username, password_hash, role, created_at) VALUES ('u-legacy', 'legacy-user', 'x', 'user', '2026-01-01')")
        c.execute("INSERT INTO documents(id, filename, filetype, created_at, meta_json) VALUES ('d-legacy', 'old.pdf', 'pdf', '2026-01-01', '{}')")
        c.execute("INSERT INTO chunks(id, doc_id, chunk_index, text, source_json) VALUES ('c-legacy', 'd-legacy', 0, 'legacy text', '{}')")
        c.commit()
    monkeypatch.setenv("ECHOMIND_EDITION", "law")
    db.init_db()                                             # the Law edition upgrades in place at boot
    with closing(sqlite3.connect(fresh_db)) as c:
        assert all(r["state"] == "applied" for r in runner.status(c))
        assert c.execute("SELECT username FROM users WHERE id='u-legacy'").fetchone() == ("legacy-user",)
        assert c.execute("SELECT text FROM chunks WHERE id='c-legacy'").fetchone() == ("legacy text",)
        assert c.execute("SELECT filename FROM documents WHERE id='d-legacy'").fetchone() == ("old.pdf",)


def _kept(db) -> dict:
    with closing(sqlite3.connect(db)) as c:
        return {"documents": c.execute("SELECT id, matter_id, title, deleted_at FROM legal_documents ORDER BY id").fetchall(),
                "versions": c.execute("SELECT id, sha256 FROM legal_document_versions ORDER BY id").fetchall(),
                "chunks": c.execute("SELECT COUNT(*) FROM legal_chunks").fetchone()[0],
                "holds": c.execute("SELECT id, status FROM legal_holds ORDER BY id").fetchall(),
                "scopes": c.execute("SELECT hold_id, object_id FROM legal_hold_scopes ORDER BY hold_id").fetchall(),
                "audit": c.execute("SELECT COUNT(*), MAX(hash) FROM audit_events").fetchone()}


def test_t13_rolling_back_the_latest_upgrades_keeps_held_records(seeded, fresh_db):
    from app.legal.migrations import runner
    before = _kept(fresh_db)
    with closing(sqlite3.connect(fresh_db)) as c:
        assert runner.migrate_down(c, target=10) == [12, 11]  # connectors and exports reverted
    assert _kept(fresh_db) == before
    with closing(sqlite3.connect(fresh_db)) as c:
        assert runner.migrate_up(c) == [11, 12]
    assert _kept(fresh_db) == before
    r = held_deletion(as_user(seeded, "bob"), seeded)
    assert r.status_code == 409 and "legal hold" in r.json()["detail"]


@pytest.mark.parametrize("target,why", [(9, "active legal hold"), (5, "audit trail"), (0, "audit trail")])
def test_t13_a_rollback_that_would_delete_kept_records_is_refused(seeded, fresh_db, target, why):
    from app.legal.migrations import runner
    before = _kept(fresh_db)
    with closing(sqlite3.connect(fresh_db)) as c:
        with pytest.raises(runner.MigrationError, match=why):
            runner.migrate_down(c, target=target)
        assert all(r["state"] == "applied" for r in runner.status(c))
    assert _kept(fresh_db) == before


def test_t13_a_rollback_that_would_widen_source_permissions_is_refused(seeded, fresh_db):
    from app.legal.migrations import runner
    with closing(sqlite3.connect(fresh_db)) as c:
        c.execute("UPDATE legal_documents SET acl_restricted = 1 WHERE id = ?", (seeded["memo"]["id"],))
        c.commit()
        with pytest.raises(runner.MigrationError, match="restricted by their source"):
            runner.migrate_down(c, target=11)
        assert all(r["state"] == "applied" for r in runner.status(c))
