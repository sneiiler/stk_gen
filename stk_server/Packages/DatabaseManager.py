# -*- coding: utf-8 -*-
# ***************************************
# * Author      : yinkaifeng
# * Email       : yinkaifeng@cast.casc
# * Create Time : 2024-6-12 09:27:27
# * Description ：实现数据库的操作
# - Wuzhipeng Update 2024/11/27 新增方法
# ***************************************
import sqlite3
import json
import datetime
from typing import List, Tuple, Optional
from icecream import ic
import Tools

class DatabaseManager:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self._create_table()

    def _create_table(self):
        """
        创建数据表
        Returns:

        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS target_list(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    longitude REAL NOT NULL,
                    latitude REAL NOT NULL,
                    priority INTEGER NOT NULL,
                    access_period TEXT,
                    access REAL,
                    value REAL,
                    last_access_date TEXT,
                    freeze_timestamp INT,
                    target_status INTEGER,
                    update_at TEXT
                )
            """)
            conn.commit()

    def add_record(self, target_list: List):
        """
        增加一条记录
        :param target_list[List]:[(, , , , ,),(),]
                    name: target_name
                    description: 描述
                    longitude: 纬度
                    latitude: 经度
                    priority: 预定义优先级
                    access_period: 预定义重访周期
                    access: 观测分辨率
                    value: 计算价值
                    last_access_date: 上次访问日期
                    freeze_timestamp: 截止该时间戳前目标不可访问
                    target_status:
                    update_at: 数据更新时间
        :return: None
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT OR IGNORE INTO target_list(name,description, longitude, latitude, priority,access_period,
                        access,value,last_access_date,freeze_timestamp, target_status, update_at) 
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, target_list)
            conn.commit()

    def update_record(self, target_name: str, **update_values):
        """
        更新指定字段
        :param target_name:
        :param update_values: value=2.5, freeze_timestamp=1457933268, ...
        :return:
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            set_clause = ", ".join([f"{column} = ?" for column in update_values])
            set_clause = set_clause + ",update_at=?"

            update_query = f"UPDATE target_list SET {set_clause} WHERE name='{target_name}'"
            params = list(update_values.values())
            update_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            params.append(update_at)
            result = cursor.execute(update_query, params)
            conn.commit()

    def excute_raw_sql(self, sql: str):
        """
        执行原始的sql
        :param sql:
        :return:
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            return cursor.fetchall()

    def get_top_records_by_value_limited(self, limit: int) -> List[dict]:
        """
        从数据库中选取价值前xx的目标
        :param limit: 限制条数
        :return:
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            current_time = Tools.current_timestamp()
            query = (
                f"SELECT name,description, longitude, latitude, priority,access_period, access,value,last_access_date,"
                f"freeze_timestamp, target_status, update_at FROM target_list WHERE freeze_timestamp <= "
                f"'{current_time}' AND target_status=0 ORDER BY value DESC LIMIT {limit}")
            # print(query)
            cursor.execute(query)
            combined_records = cursor.fetchall()
            targets_list_dict = []
            for record in combined_records:
                targets_list_dict.append(
                    {
                        "name": record[0],
                        "description": record[1],
                        "longitude": record[2],
                        "latitude": record[3],
                        "priority": record[4],
                        "access_period": record[5],
                        "access": record[6],
                        "value": record[7],
                        "last_access_date": record[8],
                        "freeze_timestamp": record[9],
                        "target_status": record[10],
                        "update_at": record[11]
                    }
                )
            return targets_list_dict

    def get_top_records_by(self, order_type: str, asc: str, limit: int) -> List[dict]:
        """
        从数据库中前xx的目标

        Args:
            asc: 是增序还是降序
            order_type:
            limit: 限制条数
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            query = (
                f"SELECT name,description, longitude, latitude, priority,access_period, access,value,last_access_date,"
                f"freeze_timestamp, target_status, update_at FROM target_list ORDER BY {order_type} {asc} LIMIT {limit}")
            # print(query)
            cursor.execute(query)
            combined_records = cursor.fetchall()
            targets_list_dict = []
            for record in combined_records:
                targets_list_dict.append(
                    {
                        "name": record[0],
                        "description": record[1],
                        "longitude": record[2],
                        "latitude": record[3],
                        "priority": record[4],
                        "access_period": record[5],
                        "access": record[6],
                        "value": record[7],
                        "last_access_date": record[8],
                        "freeze_timestamp": record[9],
                        "target_status": record[10],
                        "update_at": record[11]
                    }
                )
            return targets_list_dict

    def get_db_structure(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()

            # 查询有哪些数据表
            cursor.execute("SELECT * FROM sqlite_master WHERE type = 'table';")
            table = cursor.fetchall()
            table_name = "target_list"
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

    def get_top_records_by_value_expand(self, limit: int) -> List[dict]:
        """
        从数据库中选取价值前xx的目标
        :param limit: 限制条数
        :return:
            records[List]:
                0 name,
                1 description,
                2 longitude,
                3 latitude,
                4 priority,
                5 access_period,
                6 access,
                7 value,
                8 last_access_date,
                9 freeze_timestamp,
                10 target_status,
                11 update_at
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            # current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_time = Tools.current_timestamp()
            query = (
                f"SELECT name,description, longitude, latitude, priority,access_period, access,value,last_access_date,"
                f"freeze_timestamp, target_status, update_at FROM target_list WHERE freeze_timestamp <= "
                f"'{current_time}' ORDER BY value DESC LIMIT {limit}")
            cursor.execute(query)
            top_records = cursor.fetchall()
            if not top_records:
                return top_records

            # 获取最小value
            min_value = top_records[-1][7]
            # 获取所有的包含该最小值的记录
            query = (
                f"SELECT name,description, longitude, latitude, priority,access_period, access,value,last_access_date,"
                f"freeze_timestamp, target_status, update_at FROM target_list WHERE freeze_timestamp <= "
                f"'{current_time}' AND value = {min_value} ORDER BY value DESC")
            cursor.execute(query)
            min_value_records = cursor.fetchall()

            # combine the record
            combined_records = [record for record in top_records if record[7] > min_value] + min_value_records
            targets_list_dict = []
            for record in combined_records:
                targets_list_dict.append(
                    {
                        "name": record[0],
                        "description": record[1],
                        "longitude": record[2],
                        "latitude": record[3],
                        "priority": record[4],
                        "access_period": record[5],
                        "access": record[6],
                        "value": record[7],
                        "last_access_date": record[8],
                        "freeze_timestamp": record[9],
                        "target_status": record[10],
                        "update_at": record[11]
                    }
                )
            return targets_list_dict

    def refresh_all_target(self):
        """
        更新目标的不可选时间
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            update_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(f"UPDATE target_list "
                           f"SET freeze_timestamp=0, "
                           f"target_status=0, "
                           f"last_access_date='',"
                           f"update_at='{update_at}'")
            conn.commit()

    # ------------------------ 新增方法 -----------------------------

    def Load_targets_from_Datebase(self):
        '''
            从数据库中获取未观测目标数据(target_status = 0)， 为加快代码测试效率随机选取2000个目标
        '''
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM (SELECT * FROM target_list ORDER BY priority DESC LIMIT 2000) AS top_targets "
                            "WHERE target_status = 0")
            records = cursor.fetchall()
            if not records:
                return records
        
        targets_list = []
        for row in records:
            # ic(row)
            access_data = []
            access_data_str = json.loads(row[7])
            for dic_str in access_data_str:
                dic = json.loads(dic_str)
                access_data.append(dic)
            
            targets_list.append(
                {
                    "name": row[1],
                    "description": row[2],
                    "longitude": row[3],
                    "latitude": row[4],
                    "priority": row[5],
                    "access_period": row[6],
                    "access": access_data,
                    "value": row[8],
                    "last_access_date": row[9],
                    "freeze_timestamp": row[10],
                    "target_status": row[11],
                    "update_at": row[12],
                    "access_orbit": json.loads(row[13])
                })
        
        return targets_list
            
    def update_target_status(self, str_list:list):
        '''
            已被观测的目标状态更新 target_status=1
        '''
        update_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        param = [update_at] + list(str_list)
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            query = 'UPDATE target_list SET target_status = 1, update_at = ? WHERE name IN (%s)' % ', '.join(['?'] * len(str_list))
            cursor.execute(query, param)
            conn.commit()

    def get_targets_not_visit(self) -> list:
        '''
            返回当前剩余多少未观测的目标
        '''
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            query = ("SELECT * FROM (SELECT * FROM target_list ORDER BY priority DESC LIMIT 2000) AS top_targets "
                     "WHERE target_status != 1")
            cursor.execute(query)
            results = cursor.fetchall()

        targets_not_visit = []
        for record in results:
            access_data = []
            access_data_str = json.loads(record[7])
            for dic_str in access_data_str:
                dic = json.loads(dic_str)
                access_data.append(dic)

            targets_not_visit.append(
                {
                    "name": record[1],
                    "description": record[2],
                    "longitude": record[3],
                    "latitude": record[4],
                    "priority": record[5],
                    "access_period": record[6],
                    "access": access_data,
                    "value": record[8],
                    "last_access_date": record[9],
                    "freeze_timestamp": record[10],
                    "target_status": record[11],
                    "update_at": record[12],
                    "access_orbit": json.loads(record[13])
                }
            )

        return targets_not_visit
        
    def update_orbit_cnt(self, target_name:list, orbit_cnt:int):
        '''
            更新当前已观测的轨道数
        '''
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            placeholders = ', '.join(['?'] * len(target_name))
            # 初始化 access_orbit 字段
            init_query = f"UPDATE target_list SET access_orbit = json('[]') WHERE name IN ({placeholders}) AND access_orbit IS NULL"
            cursor.execute(init_query, target_name)

            insert_query = f"UPDATE target_list SET access_orbit = json_insert(access_orbit, '$[#]', ?) WHERE name IN ({placeholders})"
            params = [orbit_cnt] + target_name
            cursor.execute(insert_query, params)
            conn.commit()

    def get_targets_with_orbit(self, orbit_cnt: int) -> List[dict]:
        '''
            获取包含指定轨道编号的所有目标
        '''
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            query = f"""
                SELECT t.*
                FROM target_list t
                JOIN json_each(t.access_orbit) AS j
                ON json_extract(j.value, '$') = ?
            """
            cursor.execute(query, (orbit_cnt,))
            results = cursor.fetchall()

        targets_list_dict = []
        for record in results:
            access_data = []
            access_data_str = json.loads(record[7])
            for dic_str in access_data_str:
                dic = json.loads(dic_str)
                access_data.append(dic)
            
            targets_list_dict.append(
                {
                    "name": record[1],
                    "description": record[2],
                    "longitude": record[3],
                    "latitude": record[4],
                    "priority": record[5],
                    "access_period": record[6],
                    "access": access_data,
                    "value": record[8],
                    "last_access_date": record[9],
                    "freeze_timestamp": record[10],
                    "target_status": record[11],
                    "update_at": record[12],
                    "access_orbit": json.loads(record[13])
                }
            )
        return targets_list_dict

    def reset_orbit_cnt(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE target_list "
                           f"SET access = json('[]')")
                        #    f"SET access_orbit = json('[]')")
            conn.commit()

    def add_column_to_db(self, column_name:str, column_type:str):
        '''
            添加新列
        '''
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            
            # 检查列是否存在
            cursor.execute(f"PRAGMA table_info(target_list)")
            columns = cursor.fetchall()
            existing_columns = [column[1] for column in columns]
            
            if column_name not in existing_columns:
                query = f"ALTER TABLE target_list ADD COLUMN {column_name} {column_type}"
                cursor.execute(query)
                conn.commit()
            else:
                print(f"列 {column_name} 已经存在，不进行添加")

    def alter_column_type(self, column_name: str, new_column_type: str):
        """
        修改数据库中某一列的数据类型

        :param table_name: 表名
        :param column_name: 要修改的列名
        :param new_column_type: 新的数据类型
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()

            # 创建临时表
            cursor.execute("PRAGMA table_info(target_list)")
            columns = cursor.fetchall()
            temp_table_name = "target_list_temp"
            create_temp_table_sql = f"CREATE TABLE {temp_table_name} ("

            for column in columns:
                if column[1] == column_name:
                    create_temp_table_sql += f"{column_name} {new_column_type}, "
                else:
                    create_temp_table_sql += f"{column[1]} {column[2]}, "

            # 去掉最后一个逗号
            create_temp_table_sql = create_temp_table_sql.rstrip(", ") + ")"
            cursor.execute(create_temp_table_sql)

            # 将数据从原表复制到临时表
            columns_names = [column[1] for column in columns]
            columns_str = ", ".join(columns_names)
            cursor.execute(f"INSERT INTO {temp_table_name} ({columns_str}) SELECT {columns_str} FROM target_list")

            # 删除原表
            cursor.execute(f"DROP TABLE target_list")

            # 将临时表重命名为原表
            cursor.execute(f"ALTER TABLE {temp_table_name} RENAME TO target_list")

            conn.commit()

    def create_idx_priority(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_priority ON target_list(priority DESC)")
            conn.commit()
