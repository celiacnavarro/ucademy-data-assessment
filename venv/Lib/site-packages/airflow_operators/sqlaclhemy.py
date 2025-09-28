from typing import Callable

from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.models import TaskInstance
from sqlalchemy.orm import sessionmaker, DeclarativeMeta, Session


class SQLAlchemySessionOperator(PythonOperator):
    def __init__(
            self, conn_id: str, python_callable: Callable, *args, **kwargs
    ):
        super().__init__(*args, python_callable=python_callable, **kwargs)
        self.conn_id = conn_id

    def get_session_factory(self) -> sessionmaker:
        hook = MySqlHook(self.conn_id)
        engine = hook.get_sqlalchemy_engine()

        return sessionmaker(bind=engine)

    def execute_callable(self):
        session_factory = self.get_session_factory()

        with session_factory() as session:
            try:
                result = self.python_callable(
                    *self.op_args, session=session, **self.op_kwargs
                )
            except Exception:
                session.rollback()
                raise
            else:
                session.commit()
        return result


class OrmMergeOperator(SQLAlchemySessionOperator):

    def __init__(
        self,
        conn_id: str,
        xcom_key: str,
        orm_model: DeclarativeMeta,
        *args,
        **kwargs,
    ):
        super().__init__(
            conn_id=conn_id,
            python_callable=self.save,
            *args,
            **kwargs,
        )

        self.xcom_key = xcom_key
        self.orm_model = orm_model

    def save(self, session: Session, ti: TaskInstance):
        import json
        from airflow_operators.utils.orm_converter import OrmConverter

        org_jsons = ti.xcom_pull(key=self.xcom_key)

        for org_json in org_jsons:
            org_data = json.loads(org_json)
            organization = OrmConverter.convert_tree_to_orm(org_data, self.orm_model)
            session.merge(organization)
