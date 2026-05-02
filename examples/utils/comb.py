import json
from itertools import product


def generate_combinations(*args):
    return list(product(*args))


def combination_per_example(example):
    # Take all completions
    completion_list = [
        step["completions"] for step in example["label"]["steps"]
    ]
    # Combine them
    combinations = generate_combinations(*completion_list)

    return combinations


def combinations_for_dataset(data, to_json=False):
    user_input = []
    for x in data:
        step_combinations = combination_per_example(x)
        for combination in step_combinations:
            current_problem = {}
            current_problem["question"] = x["question"]["problem"]
            current_problem["steps"] = []

            for step in combination:
                current_problem["steps"].append(step["text"])
            user_input.append(current_problem)

    if to_json:
        with open("combined_steps.json", "w") as json_file:
            json.dump(user_input, json_file)
    return user_input
