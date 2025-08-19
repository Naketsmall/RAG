from config.configuration import TEST_IMAGE_DIR
from src.graph_manager import GraphManager
from src.graphit import Graphit

graphit = Graphit(classes_config='../config/classes.json', relations_config='../config/relations.json')
results = graphit.find_objects('../' + TEST_IMAGE_DIR)
manager = GraphManager("bolt://localhost:7687", "neo4j", "testpassword",
                      classes_config="../config/classes.json", relations_config="../config/relations.json")
manager.clean_database()
first_pic = []
for box in results[0].boxes:
    first_pic.append({'class': graphit.yolo.names[box.cls[0].item()],
                      'bbox': box.xyxy[0]})
print('objects from first pic:', first_pic)

objects = graphit.build_from_detection(results[0].orig_img, first_pic)
print(objects)
manager.add_scene(objects)

print(manager.find_similar_objects("object where we can put dishes with christmas food"))

