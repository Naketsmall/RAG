import json

from neo4j import GraphDatabase


class DBLoader:
    """
    Класс осуществляет взаимодействие с графовой БД с векторными индексами - Neo4j.
    """

    def __init__(self, uri, user, password,
                 classes_config: str = "configs/classes.json", relations_config: str = "configs/relations.json"):

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        with open(classes_config) as f:
            self.classes_config = json.load(f)

        with open(relations_config) as f:
            self.relations_config = json.load(f)


        db_generation_queries = self.generate_cypher_from_json(self.classes_config)
        with self.driver.session() as session:
            for query in db_generation_queries:
                session.run(query)

    def generate_cypher_from_json(self, schema_json):
        classes = json.loads(schema_json) if isinstance(schema_json, str) else schema_json
        queries = []

        all_relations = set()
        for class_config in classes.values():
            all_relations.update(class_config["neighbours"]["relations"])

        if all_relations:
            rels = ", ".join("{name: '" + rel + "'}" for rel in all_relations)
            queries.append(f"UNWIND [{rels}] AS rel MERGE (:RelationType {{name: rel.name}})")

        for class_name, class_config in classes.items():
            queries.append(
                "MERGE (c:Class {name: '" + class_name + "', detectable: " + str(class_config['detectable']).lower()
                + "})"
            )

            for feature_name, feature_config in class_config["features"].items():
                props = {
                    k: f"'{v}'" if isinstance(v, str) else str(v).lower() if isinstance(v, bool) else v
                    for k, v in feature_config.items()
                    if v is not None
                }
                queries.append(f"""
                    MATCH (c:Class {{name: '{class_name}'}})
                    MERGE (f:Feature {{name: '{feature_name}'}})
                    MERGE (c)-[r:HAS_FEATURE]->(f)
                    SET r += {{{', '.join(f'{k}: {v}' for k, v in props.items())}}}
                """.strip())

            max_count = class_config["neighbours"]["max_count"]
            queries.append(f"""
                MATCH (c:Class {{name: '{class_name}'}})
                SET c.max_neighbours = {max_count}
            """.strip())

            rels = class_config["neighbours"]["relations"]
            if rels:
                queries.append(f"""
                    MATCH (c:Class {{name: '{class_name}'}})
                    UNWIND {rels} AS rel
                    MATCH (rt:RelationType {{name: rel}})
                    MERGE (c)-[:CAN_HAVE_RELATIONSHIP]->(rt)
                """.strip())

        return queries

    def clean_database(self):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                DETACH DELETE n
                RETURN count(n) AS deletedNodes
            """)
            deleted_count = result.single()["deletedNodes"]
            print(f"\nУдалено {deleted_count} узлов и всех их связей")
            return deleted_count