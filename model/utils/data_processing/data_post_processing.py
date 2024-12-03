import json
import os


def expand_prompt_squads(input_file, output_file):
    """
    Expand the JSONL dataset where each record has multiple squads into individual records with one squad each.

    :param input_file: Path to the input JSONL file.
    :param output_file: Path to the output JSONL file.
    """
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            try:
                # Parse the JSON line
                data = json.loads(line)

                # Extract prompt and squad names
                prompt = data.get("prompt", "")
                squads = data.get("completion", "").split(",")  # Split squads on commas

                # Write each squad as a separate record
                for squad in squads:
                    squad_name = squad.strip()
                    if squad_name:  # Avoid empty names
                        expanded_entry = {"prompt": prompt, "squad": squad_name}
                        outfile.write(json.dumps(expanded_entry) + "\n")
            except Exception as e:
                print(f"Error processing line: {line}\n{e}")

    print(f"Data expanded and saved to {output_file}")


def main():
    """
    Main function to transform the dataset.
    """
    # Paths to input and output files
    input_file = "model/data/training_data/training_data.jsonl"
    output_file = "model/data/training_data/expanded_training_data.jsonl"

    expand_prompt_squads(input_file, output_file)


if __name__ == "__main__":
    main()
