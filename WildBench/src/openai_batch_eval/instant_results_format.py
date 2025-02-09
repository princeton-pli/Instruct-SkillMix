import json 
import sys 
import jsonlines
import os

def main():

    result_file = sys.argv[1] # a jsonl file

    MODE = "pairwise" # or "score"

    if "score" in result_file:
        MODE = "score"

    # already has been processed
    raw_file = result_file.replace(".json", ".raw.json")
    if os.path.exists(raw_file): return

    # load json file into result_file
    with open(result_file, "r") as f:
        results = json.load(f)

    # print(json.dumps(results[0], indent=2))
    results_json = []
    for item in results:
        eval_output_parsed = item["parsed_result"]
        # try:
        #     eval_output_parsed = json.load(eval_output)
        # except Exception as e:
        #     print(f"Error parsing eval_output.")
        #     print(eval_output)
        #     print(e)
        #     # eval_output_parsed = eval_output
        #     continue
        results_item = {
            "session_id": item["session_id"],
            "parsed_result": eval_output_parsed,
            "primary_tag": item["primary_tag"]
        }
        if MODE == "pairwise":
            model_A = item["assignment"]["A"]
            model_B = item["assignment"]["B"]
            # reason = eval_output_parsed["reason"]
            if "choice" not in eval_output_parsed:
                print(f"Error: choice not found in eval_output_parsed.")
                continue
            choice = eval_output_parsed["choice"]
            winner = "tie"
            if choice == "A=B":
                winner = "tie"
                extent = 0 
            elif choice == "A+":
                winner = model_A
                extent = 1
            elif choice == "A++":
                winner = model_A
                extent = 2
            elif choice == "B+":
                winner = model_B
                extent = 1
            elif choice == "B++":
                winner = model_B
                extent = 2
            else:
                print(f"Error: choice {choice} not recognized.")
                continue
            results_item.update({
                "model_A": model_A,
                "model_B": model_B,
                "winner": winner,
                "extent": extent,
            })
            
            model_A_output = item["ref_output"] if model_A == item["ref_generator"] else item["model_output"][0]
            model_B_output = item["ref_output"] if model_B == item["ref_generator"] else item["model_output"][0]
            results_item["model_outputs"] = {
                model_A: model_A_output.strip(),
                model_B: model_B_output.strip(),
            }
        elif MODE == "score":
            model_test = item["generator"]
            if "score" not in eval_output_parsed:
                print(f"Error: score not found in eval_output_parsed.")
                continue
            score  = eval_output_parsed["score"]
            results_item.update({
                "model_test": model_test,
                "score": score,
            })
            model_output = item["model_output"][0]
            results_item["model_output"] = model_output.strip()

        
        
        
        
        results_json.append(results_item)

    # write to a json file
    with open(raw_file, "w") as f:
        json.dump(results, f, indent=2)

    with open(result_file, "w") as f:
        json.dump(results_json, f, indent=2)
        print(f"Output file written to {result_file}")

if __name__ == "__main__":
    main()