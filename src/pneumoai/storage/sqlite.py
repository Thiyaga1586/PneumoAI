from sqlalchemy import create_engine

_engine = None

def init_db():
    global _engine

    from pneumoai.common.settings import settings

    db_url = settings.database_url

    if not db_url:
        raise RuntimeError("DATABASE_URL not set")

    _engine = create_engine(db_url, pool_pre_ping=True)

    return _engine