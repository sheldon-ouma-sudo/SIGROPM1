import json
import os

# Dictionary to hold the combined data
combined_data = {}


def combine_json_files():
    """Combine JSON files from the base_data directory."""
    # Reference the correct path to the base_data directory
    base_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "base_data"
    )

    # File mappings
    file_key_mapping = {
        "culture_data.json": "culture",
        "expanded_sports_data.json": "sports",
        "politics_interest_data.json": "politics",
        "science_data.json": "science",
        "social.json": "social",
        "wellness_data.json": "wellness",
    }

    for file_name, key in file_key_mapping.items():
        file_path = os.path.join(base_data_dir, file_name)
        print(f"Attempting to read file: {file_path}")

        if os.path.exists(file_path):
            print(f"File {file_path} found. Reading now...")
            with open(file_path, "r") as file:
                try:
                    data = json.load(file)
                    print(f"Loaded data from {file_name}: {str(data)[:100]}...")
                    combined_data[key] = data
                except json.JSONDecodeError as e:
                    print(f"Error reading {file_path}: {e}. Skipping this file.")
        else:
            print(f"File {file_path} does not exist. Skipping.")

    print("Combined data successfully loaded.")
    return combined_data


def save_combined_data(output_file="combined_data.json"):
    """Save the combined data to the base_data directory."""
    data = combine_json_files()

    # Save the combined data in the base_data directory
    base_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "base_data"
    )
    output_path = os.path.join(base_data_dir, output_file)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Combined data saved to {output_path}")


def load_combined_data():
    """Load the combined data from combined_data.json in base_data."""
    base_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "base_data"
    )
    file_path = os.path.join(base_data_dir, "combined_data.json")

    with open(file_path, "r") as f:
        return json.load(f)


def main():
    save_combined_data()


if __name__ == "__main__":
    main()