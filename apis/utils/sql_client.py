"""
SQLite client for Fooocus API
"""
import os
import time
from datetime import datetime
import copy

from sqlalchemy import Integer, Float, VARCHAR, JSON, create_engine
from sqlalchemy.orm import declarative_base, Session, Mapped, mapped_column


DB_PATH = os.path.join("outputs", "db.sqlite3")
Base = declarative_base()


class GenerateRecord(Base):
    """
    GenerateRecord

    __tablename__ = 'generate_record'
    """

    __tablename__ = "generate_record"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True, comment="主键")

    task_id: Mapped[str] = mapped_column(VARCHAR(255), nullable=False, comment="Task ID")
    req_params: Mapped[dict] = mapped_column(JSON, nullable=False, comment="Request Parameters")
    in_queue_mills: Mapped[int] = mapped_column(Integer, nullable=False, comment="In Queue Milliseconds")
    start_mills: Mapped[int] = mapped_column(Integer, nullable=False, comment="Start Milliseconds")
    finish_mills: Mapped[int] = mapped_column(Integer, nullable=False, comment="Finish Milliseconds")
    task_status: Mapped[str] = mapped_column(VARCHAR(255), nullable=False, comment="Task Status")
    progress: Mapped[float] = mapped_column(Float, nullable=False, comment="Progress")
    webhook_url: Mapped[str] = mapped_column(VARCHAR(255), nullable=True, comment="Webhook URL")
    result: Mapped[list] = mapped_column(JSON, nullable=True, comment="Result")


engine = create_engine(DB_PATH)

session = Session(engine)
Base.metadata.create_all(engine, checkfirst=True)
session.close()
