from config.configuration import TEST_IMAGE_DIR
from src.graphit import Graphit
from src.prompt_manager import PromptManager

graphit = Graphit(classes_config='../config/classes.json', relations_config='../config/relations.json')
manager = PromptManager()


results = graphit.find_objects('../' + TEST_IMAGE_DIR)

first_pic = []
for box in results[0].boxes:
    first_pic.append({'class': graphit.yolo.names[box.cls[0].item()],
                      'bbox': box.xyxy[0]})
print('objects from first pic:', first_pic)

objects = [{"id": obj.id,
            "class_name": obj.class_name,
            "features": obj.features,
            "neighbours": obj.neighbours} for obj in graphit.build_from_detection(results[0].orig_img, first_pic)]



prompt_name = "prompt_1"
input_data = {
    "db_data": objects,
    "user_query": "Найди объекты, на которых гипотетически можно поиграть в компьютерную игру DOOM"
}

full_prompt = manager.get_prompt(prompt_name, input_data)
print(full_prompt)
print(len(full_prompt), "\n")

response = graphit.llm.chat(full_prompt)
print(response)
with open('../llm_responses/TLDBWLLM_2_Max.txt', 'w') as f:
    f.write(str(response.choices[0].message.content))
