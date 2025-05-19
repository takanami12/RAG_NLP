from llm_services import LLMService
from dotenv import load_dotenv
import os
import sys
import json

load_dotenv()


MODEL_EMBEDDING = os.getenv("MODEL_EMBEDDING")
CHUNK_SIZE = 256
PATH_TEST_DATA = "../data/demo_questions.json"

model_list = [
    {"name": "meta-llama-3.2", "link": "meta-llama/Llama-3.2-3B-Instruct"},
    {"name": "qwen-1.5", "link": "Qwen/Qwen1.5-0.5B-Chat"},
    {"name": "gemma-3.4", "link": "google/gemma-3-4b-it"},
]


def main(model_index):
    load_dotenv()

    model_variable = model_list[model_index]

    llm_service = LLMService(
        # You can specify a different model from the list if needed
        model_name=model_variable["link"],
        use_rag=True,
        model_embedding=MODEL_EMBEDDING,
    )

    # Load input JSON
    with open(PATH_TEST_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"--- Loaded {len(data)} questions from {PATH_TEST_DATA} ---")

    # Process each question
    for i, item in enumerate(data):
        prompt = item["question"]
        print(f"\n--- Query {i + 1}: {prompt} ---")
        answer = llm_service.generate_text(prompt, max_length=128)
        print(f"rag_answer {i + 1}:", answer["rag_answer"])
        item["rag_prompt"] = answer["rag_prompt"]
        item["rag_answer"] = answer["rag_answer"]

    print("\n--- All queries processed ---")

    # Save to new JSON file
    # json_save_path = "/data/{}-result.json".format(model_variable["name"])
    # with open(json_save_path, "w", encoding="utf-8") as f:
    #     json.dump(data, f, ensure_ascii=False, indent=2)

    # Save to new JSON file
    json_save_path = f"../data/{model_variable['name']}-result.json"
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # os.makedirs("stdout", exist_ok=True)
    # sys.stdout = open("stdout/main_out.txt", "w")
    # sys.stderr = open("stdout/main_err.txt", "w")

    # main(model_index=1)

    # sys.stdout.close()
    # sys.stderr.close()

    os.makedirs("stdout", exist_ok=True)
    out_file = open("stdout/main_out.txt", "w", encoding="utf-8")
    err_file = open("stdout/main_err.txt", "w", encoding="utf-8")
    sys.stdout = out_file
    sys.stderr = err_file

    try:
        main(model_index=0)
    except Exception as e:
        print(f"Exception occurred: {e}", file=sys.stderr)
        raise
    finally:
        out_file.flush()
        err_file.flush()
        out_file.close()
        err_file.close()