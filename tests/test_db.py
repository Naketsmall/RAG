from src.graph_manager import GraphManager

"""
Скрипт проверяет взаимодествие программы с БД
Можно открыть в браузере http://localhost:7474/ и посмотреть визуализацию БД.

Перед запуском скрипта нужно запустить контейнер с БД из docker_scripts/run_container.sh
"""

loader = GraphManager("bolt://localhost:7687", "neo4j", "testpassword",
                      classes_config="../config/classes.json", relations_config="../config/relations.json")
#loader.clean_database()