from src.graph_manager import GraphManager


loader = GraphManager("bolt://localhost:7687", "neo4j", "testpassword", classes_config="../config/classes.json", relations_config="../config/relations.json")
#loader.clean_database()