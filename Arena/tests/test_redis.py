import config
import redis


def test_redis_exists():
    r = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=0)
    try:
        r.ping()
    except Exception as exc:
        print(f'Unable to connect to Redis in {config.REDIS_HOST}:{config.REDIS_PORT}')
        config.IS_USE_REDIS = False
        config.IS_AGENT_ENABLED = False
