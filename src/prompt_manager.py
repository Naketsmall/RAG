import json


class PromptManager:
    def __init__(self, file_path="../config/prompts.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            self.prompts = json.load(f)

    def get_prompt(self, name, params=None):
        prompt_data = self.prompts.get(name)
        if not prompt_data:
            raise ValueError(f"Prompt '{name}' not found")


        parts = [
            f"# Системная роль\n{prompt_data['role']}",
            f"\n# Задача\n{prompt_data['task_description']}",
            f"\n# Ограничения\n" + "\n".join(f"- {c}" for c in prompt_data['constraints']),
            f"\n# Входные данные\n{json.dumps(params, ensure_ascii=False)}",
            f"\n# Требуемый формат вывода\n{json.dumps(prompt_data['output_format'], indent=2)}"
        ]

        # Добавляем примеры
        if "examples" in prompt_data:
            parts.append("\n# Примеры использования")
            for i, example in enumerate(prompt_data['examples'], 1):
                parts.append(f"\nПример {i}:\nВход: {json.dumps(example['input'], ensure_ascii=False)}")
                parts.append(f"Ожидаемый вывод: {json.dumps(example['output'], ensure_ascii=False)}")

        return "\n".join(parts)


