from flow.ket_partner.config import KetConfig, load_config


def test_load_config_returns_defaults():
    cfg = load_config()
    assert isinstance(cfg, KetConfig)
    assert cfg.vocab_refill.low_watermark == 5
    assert cfg.vocab_refill.high_watermark == 10
    assert cfg.vocab_refill.interval_turns == 5
    assert cfg.summary.interval_turns == 15
    assert cfg.validate_retry_limit == 2
