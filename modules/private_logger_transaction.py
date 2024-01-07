# Copyright 2023 SLAPaper
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import typing as tg

class LoggerTransactionManager:
    """sequential logging"""
    _transaction = False

    # filepath: content
    _cache: dict[str, str] = {}

    @staticmethod
    def is_transaction() -> bool:
        return LoggerTransactionManager._transaction

    @staticmethod
    def begin() -> None:
        LoggerTransactionManager._transaction = True

    @staticmethod
    def flush() -> None:
        LoggerTransactionManager._transaction = False
        for filepath, content in LoggerTransactionManager._cache.items():
            with open(filepath, "w", encoding='utf8') as f:
                f.write(content)
            print(f'Private log flushing to: {filepath}')

        LoggerTransactionManager._cache.clear()

    @staticmethod
    def push(filepath: str, content: str) -> None:
        LoggerTransactionManager._cache[filepath] = content

