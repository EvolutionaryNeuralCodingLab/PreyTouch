import config
import redis
import pytest


def test_redis_exists():
    r = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=0)
    try:
        r.ping()
    except Exception as exc:
        pytest.fail(f'Unable to connect to Redis in {config.REDIS_HOST}:{config.REDIS_PORT}')
        # config.IS_USE_REDIS = False
        # config.IS_AGENT_ENABLED = False
