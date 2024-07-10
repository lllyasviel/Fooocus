"""
SQLite client for Fooocus API
"""
import copy
import json
import os

from sqlalchemy import Integer, Float, VARCHAR, JSON, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Mapped, mapped_column
from modules.config import path_outputs


DB_PATH = os.path.join(path_outputs, "db.sqlite3")
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
    in_queue_mills: Mapped[int] = mapped_column(Integer, nullable=True, comment="In Queue Milliseconds")
    start_mills: Mapped[int] = mapped_column(Integer, nullable=True, comment="Start Milliseconds")
    finish_mills: Mapped[int] = mapped_column(Integer, nullable=True, comment="Finish Milliseconds")
    task_status: Mapped[str] = mapped_column(VARCHAR(255), nullable=True, comment="Task Status")
    progress: Mapped[float] = mapped_column(Float, nullable=True, comment="Progress")
    webhook_url: Mapped[str] = mapped_column(VARCHAR(255), nullable=True, comment="Webhook URL")
    result: Mapped[list] = mapped_column(JSON, nullable=True, comment="Result")

    def __repr__(self):
        d = copy.deepcopy(self.__dict__)
        d.pop("_sa_instance_state")
        return json.dumps(d, ensure_ascii=False)


engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False}, future=True)

Base.metadata.create_all(engine, checkfirst=True)
Session = sessionmaker(bind=engine)
session = Session()
session.close()
