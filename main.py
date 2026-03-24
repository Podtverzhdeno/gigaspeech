import uuid
import time
from graph import build_graph
from config import speaker_bio, inputs
from llm import giga, chatgpt

def main():
    graph = build_graph()

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    start = time.time()

    print(" Запуск генерации речи...\n")

    for output in graph.stream(inputs, config=config, stream_mode="updates"):
        current_agent = next(iter(output))
        print(f"Отработал агент: {current_agent} | Время: {int(time.time() - start)}с")

    final_state = graph.get_state(config=config).values
    speech = final_state.get("result_speech", "")
    speech = speech.replace("```markdown", "").replace("```", "").strip()

    print("\n" + "="*60)
    print(" ФИНАЛЬНАЯ РЕЧЬ:")
    print("="*60 + "\n")
    print(speech)

    # Сохраняем результат
    with open("result_speech.md", "w", encoding="utf-8") as f:
        f.write(speech)
    print("\n Речь сохранена в result_speech.md")

if __name__ == "__main__":
    main()