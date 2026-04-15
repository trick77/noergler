import logging

import asyncpg

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def create_pool(dsn: str) -> asyncpg.Pool:
    global _pool
    pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
    assert pool is not None
    _pool = pool
    async with pool.acquire() as conn:
        version = await conn.fetchval("SELECT version()")
    logger.info("PostgreSQL connection pool created (%s)", version)
    return pool


def get_pool() -> asyncpg.Pool | None:
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL connection pool closed")
