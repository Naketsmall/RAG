import json
from typing import List

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from src.scene_object import SceneObject

class GraphManager:
    """
    Класс осуществляет взаимодействие с графовой БД с векторными индексами - Neo4j.
    """

    def __init__(self, uri, user, password,
                 classes_config: str = "configs/classes.json", relations_config: str = "configs/relations.json"):

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.model = SentenceTransformer('../../all-MiniLM-L6-v2')

        with open(classes_config) as f:
            self.classes_config = json.load(f)

        with open(relations_config) as f:
            self.relations_config = json.load(f)

        db_generation_queries = self.generate_db_from_json(self.classes_config)
        with self.driver.session() as session:
            for query in db_generation_queries:
                session.run(query)


            session.run("""
                CREATE VECTOR INDEX object_embedding_index IF NOT EXISTS
                FOR (o:Object) ON (o.embedding)
                OPTIONS { indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
            """)

    def generate_db_from_json(self, schema_json):
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

    def add_scene(self, objects: List[SceneObject]):
        """
        Добавляет сцену (список объектов) в графовую БД:
        - создаёт узлы объектов с эмбеддингами и признаками
        - связывает объекты с их классами
        - создаёт рёбра отношений (из neighbours)
        """
        with self.driver.session() as session:
            for obj in objects:
                obj_embedding = self.model.encode(obj.get_semantic_repr()).tolist()
                props = {
                    "id": obj.id,
                    "class_name": obj.class_name,
                    "bbox": obj.bbox.tolist() if hasattr(obj.bbox, "tolist") else list(obj.bbox),
                    "embedding": obj_embedding,
                }
                props.update(obj.features)
                session.run(
                    """
                    MERGE (o:Object {id: $id})
                    SET o += $props
                    WITH o
                    MATCH (c:Class {name: $class_name})
                    MERGE (o)-[:INSTANCE_OF]->(c)
                    """,
                    id=obj.id,
                    class_name=obj.class_name,
                    props=props
                )

            for obj in objects:
                for rel_type, neighbours in obj.neighbours.items():
                    for nb_id in neighbours:
                        session.run(
                            """
                            MATCH (o1:Object {id: $id1})
                            MATCH (o2:Object {id: $id2})
                            MERGE (o1)-[r:RELATION]->(o2)
                            SET r.type = $rel_type
                            """,
                            id1=obj.id,
                            id2=nb_id,
                            rel_type=rel_type
                        )

    def find_similar_objects(self, query: str, top_k: int = 5):
        """
        Находит объекты в базе, похожие на текстовый запрос (по эмбеддингам).
        """
        query_embedding = self.model.encode(query).tolist()

        with self.driver.session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes('object_embedding_index', $top_k, $embedding)
                YIELD node, score
                RETURN node.id AS id, node.class_name AS class, score
                """,
                embedding=query_embedding,
                top_k=top_k
            )
            return result.data()

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
