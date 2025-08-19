from src.prompt_manager import PromptManager

manager = PromptManager()

prompt_name = "prompt_1"
input_data = {
    "db_data": [2, 3, 4],
    "user_query": "Найди объекты связанные с AI"
}

full_prompt = manager.get_prompt(prompt_name, input_data)
print(full_prompt)
print(len(full_prompt))
#response = llm.generate(full_prompt)