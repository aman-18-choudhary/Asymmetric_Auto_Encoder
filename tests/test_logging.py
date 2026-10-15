import logging

from asymmetric_autoencoder.utils.logging import get_logger, setup_logging


def test_setup_is_idempotent_and_writes_file(tmp_path):
    log_file = tmp_path / "logs" / "run.log"
    setup_logging(log_file=log_file)
    logger = setup_logging(log_file=log_file)
    assert len(logger.handlers) == 2
    get_logger("tests.demo").info("hello-phase-1")
    for handler in logger.handlers:
        handler.flush()
    assert "hello-phase-1" in log_file.read_text()
    setup_logging(level=logging.INFO)


def test_get_logger_namespacing():
    assert get_logger("foo").name == "asymmetric_autoencoder.foo"
    assert get_logger("asymmetric_autoencoder.bar").name == "asymmetric_autoencoder.bar"
