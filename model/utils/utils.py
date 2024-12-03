import json
import os

# Dictionary to hold the combined data
combined_data = {}

def get_base_data_dir():
    """Return the correct path to the base_data directory."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "base_data")

def combine_json_files():
    """Combine JSON files from the base_data directory."""
    base_data_dir = get_base_data_dir()

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
            try:
                with open(file_path, "r") as file:
                    data = json.load(file)
                    print(f"Loaded data from {file_name}.")
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
    base_data_dir = get_base_data_dir()

    # Create base_data directory if it doesn't exist
    if not os.path.exists(base_data_dir):
        os.makedirs(base_data_dir)
        print(f"Created directory {base_data_dir}.")

    output_path = os.path.join(base_data_dir, output_file)
    try:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Combined data saved to {output_path}")
    except IOError as e:
        print(f"Error saving combined data: {e}")

def load_combined_data():
    """Load the combined data from combined_data.json in base_data."""
    base_data_dir = get_base_data_dir()
    file_path = os.path.join(base_data_dir, "combined_data.json")

    if not os.path.exists(file_path):
        print(f"Combined data file not found at {file_path}.")
        return {}

    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error loading combined data: {e}")
        return {}

def main():
    save_combined_data()

if __name__ == "__main__":
    main()
