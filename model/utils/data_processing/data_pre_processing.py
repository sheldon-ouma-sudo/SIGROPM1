import json
import os
import re
from utils import load_combined_data


class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = {}
        self.api_query_child = {
            "details": "No API query data yet"
        }  # Initialize api_query_child

        # Add 'details' as a separate child node only if this node isn't itself 'details'
        if name != "details":
            self.children["details"] = TreeNode("details")

    def add_child(self, child_name):
        """Add a child node if it doesn't already exist."""
        if child_name not in self.children:
            self.children[child_name] = TreeNode(child_name)
        return self.children[child_name]

    def set_api_query_child(self, details):
        """Set data in the 'details' child node."""
        self.children["details"] = TreeNode("details")
        self.children["details"].children = {"details": details}

    def ensure_api_query_child(self):
        """Ensure that api_query_child is initialized if it's missing."""
        if not self.api_query_child:
            self.api_query_child = {"details": "No API query data yet"}

    def to_dict(self):
        """Convert the TreeNode to a dictionary for JSON output."""
        return {
            "name": self.name,
            "api_query_child": self.api_query_child,
            "children": {
                name: child.to_dict() for name, child in self.children.items()
            },
        }

    def __repr__(self):
        return self._print_tree()

    def _print_tree(self, level=0):
        indent = "    " * level
        result = f"{indent}{self.name}\n"
        result += f"{indent}    apiQuery: {self.api_query_child}\n"
        if self.children:
            result += f"{indent}    children:\n"
            for child in self.children.values():
                result += child._print_tree(level + 1)
        return result


def clean_query(query):
    """Remove non-essential words like 'e.g.' from the query."""
    stop_words = {"and", "e.g.", "or", "of", "the"}
    words = [word.strip() for word in re.split(r"\s+", query) if word.strip()]
    cleaned_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(cleaned_words)


def process_node_name(node_name):
    """Extract main category and subcategories, removing non-essential words."""
    main_category = node_name.split("(")[0].strip()
    subcategories = re.findall(r"\((.*?)\)", node_name)
    cleaned_subcategories = []
    if subcategories:
        for subcat in subcategories:
            subcat_items = [clean_query(item).strip() for item in subcat.split(",")]
            cleaned_subcategories.extend([item for item in subcat_items if item])
    return main_category, cleaned_subcategories


def recursively_add_nodes(node, data):
    """Recursively add nodes to the tree, adding a 'details' child for each node."""
    if isinstance(data, dict):
        for key, value in data.items():
            main_category, subcategories = process_node_name(key)
            # Add main category node
            child_node = node.add_child(main_category)

            # Add each subcategory within parentheses as a child node
            for subcategory in subcategories:
                child_node.add_child(subcategory)

            recursively_add_nodes(child_node, value)
            child_node.ensure_api_query_child()  # Ensure api_query_child is set
    elif isinstance(data, list):
        for item in data:
            main_category, subcategories = process_node_name(item)
            leaf_node = node.add_child(main_category)

            # Add subcategories as children if present
            for subcategory in subcategories:
                leaf_node.add_child(subcategory)

            leaf_node.ensure_api_query_child()  # Ensure api_query_child is set
    else:
        # Add details to the 'details' child node for leaf nodes
        node.children["details"] = TreeNode("details")
        node.children["details"].children = {"details": data}
        node.ensure_api_query_child()  # Ensure api_query_child is set


def build_tree_from_json(data):
    """Build a tree structure from JSON data."""
    root = TreeNode("Root")
    for key, value in data.items():
        child_node = root.add_child(key)
        recursively_add_nodes(child_node, value)
        child_node.ensure_api_query_child()
    return root


def load_and_build_trees():
    """Load the combined data and build trees for each section."""
    data = load_combined_data()
    trees = {}
    for section in ["culture", "sports", "politics", "science", "social", "wellness"]:
        if section in data:
            trees[section] = build_tree_from_json(data[section])
    return trees


def save_trees_to_json(trees, directory="base_data"):
    """Save each tree to a JSON file in the base_data directory."""
    base_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), directory)

    if not os.path.exists(base_data_dir):
        os.makedirs(base_data_dir)

    for tree_name, tree in trees.items():
        tree_file = os.path.join(base_data_dir, f"{tree_name}_tree.json")
        with open(tree_file, "w") as f:
            json.dump(tree.to_dict(), f, indent=4)
        print(f"Saved {tree_name} tree to {tree_file}")


def save_entire_tree_to_json(trees, directory="base_data", filename="entire_tree.json"):
    """Combine all section trees under a single root and save to a JSON file in base_data."""
    base_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), directory)

    if not os.path.exists(base_data_dir):
        os.makedirs(base_data_dir)

    # Create the root for the entire tree and attach section trees
    root = TreeNode("Root")
    for section_name, tree in trees.items():
        root.children[section_name] = tree  # Attach each section tree under the root

    # Save the entire tree to the specified path
    entire_tree_path = os.path.join(base_data_dir, filename)
    with open(entire_tree_path, "w") as f:
        json.dump(root.to_dict(), f, indent=4)
    print(f"Entire tree structure saved to {entire_tree_path}")


# Build the trees and store them in a global variable
TREES = load_and_build_trees()

if __name__ == "__main__":
    # Print the trees for verification
    for tree_name, tree in TREES.items():
        print(f"\n{tree_name.capitalize()} Tree:")
        print(tree)

    # Save individual trees to JSON files in base_data
    save_trees_to_json(TREES)

    # Save the entire tree structure to a single JSON file in base_data
    save_entire_tree_to_json(TREES)

# Build the trees and store them in a global variable
TREES = load_and_build_trees()

if __name__ == "__main__":
    # Print the trees for verification
    for tree_name, tree in TREES.items():
        print(f"\n{tree_name.capitalize()} Tree:")
        print(tree)

    # Save individual trees to JSON files
    save_trees_to_json(TREES)

    # Save the entire tree structure to a single JSON file
    save_entire_tree_to_json(TREES)
