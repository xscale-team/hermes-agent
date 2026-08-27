"""Regression coverage for durable compression-timeout escalation.

A compressor is routinely rebuilt while a long Desktop session survives.  The
host timeout ladder must therefore belong to the durable session, not to one
``ContextCompressor`` Python object.  Otherwise every rebuild returns to the
first cooldown rung and an unhealthy session can spend ten minutes compressing,
one minute "cooling down", and then repeat forever.
"""

from pathlib import Path
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from hermes_state import SessionDB


def _compressor(db: SessionDB, session_id: str) -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
    compressor.bind_session_state(db, session_id)
    return compressor


def _db(tmp_path: Path) -> SessionDB:
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="cli")
    return db


def test_timeout_ladder_survives_compressor_reconstruction(tmp_path):
    db = _db(tmp_path)

    with patch("agent.context_compressor.time.time", return_value=1_000.0):
        _compressor(db, "s1").record_timeout_failure("first timeout")
    first = db.get_compression_failure_cooldown_row("s1")
    assert first["timeout_count"] == 1
    assert first["cooldown_until"] == 1_060.0

    # A brand-new object must atomically advance the same session's ladder.
    with patch("agent.context_compressor.time.time", return_value=2_000.0):
        _compressor(db, "s1").record_timeout_failure("second timeout")
    second = db.get_compression_failure_cooldown_row("s1")
    assert second["timeout_count"] == 2
    assert second["cooldown_until"] == 2_300.0

    with patch("agent.context_compressor.time.time", return_value=3_000.0):
        _compressor(db, "s1").record_timeout_failure("third timeout")
    third = db.get_compression_failure_cooldown_row("s1")
    assert third["timeout_count"] == 3
    assert third["cooldown_until"] == 3_900.0


def test_successful_compression_resets_durable_timeout_ladder(tmp_path):
    db = _db(tmp_path)
    compressor = _compressor(db, "s1")

    with patch("agent.context_compressor.time.time", return_value=1_000.0):
        compressor.record_timeout_failure("timeout")
    assert db.get_compression_failure_cooldown_row("s1")["timeout_count"] == 1

    compressor._clear_compression_failure_cooldown()
    cleared = db.get_compression_failure_cooldown_row("s1")
    assert cleared["timeout_count"] == 0
    assert cleared["cooldown_until"] is None
    assert cleared["error"] is None
