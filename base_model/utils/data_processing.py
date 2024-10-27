import json
import os
from utils import load_combined_data


class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = {}
        self.api_query_child = {"details": "No API query data yet"}

    def add_child(self, child_name):
        if child_name not in self.children:
            self.children[child_name] = TreeNode(child_name)
        return self.children[child_name]

    def set_api_query_child(self, details):
        self.api_query_child = details

    def ensure_api_query_child(self):
        if not self.api_query_child:
            self.api_query_child = {"details": "No API query data yet"}

    def to_dict(self):
        """Convert the TreeNode to a dictionary."""
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


def build_tree_from_json(data):
    """Builds a tree from JSON data."""
    root = TreeNode("Root")
    for key, value in data.items():
        child_node = root.add_child(key)
        recursively_add_nodes(child_node, value)
        child_node.ensure_api_query_child()
    return root


def recursively_add_nodes(node, data):
    """Recursively adds nodes to the tree."""
    if isinstance(data, dict):
        for key, value in data.items():
            child_node = node.add_child(key)
            recursively_add_nodes(child_node, value)
            child_node.ensure_api_query_child()
    elif isinstance(data, list):
        for item in data:
            leaf_node = node.add_child(item)
            leaf_node.ensure_api_query_child()
    else:
        node.set_api_query_child({"details": data})


def load_and_build_trees():
    """Loads the combined data and builds trees for each section."""
    data = load_combined_data()  # Load combined data from utils.py
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


# Build the trees and store them in a global variable
TREES = load_and_build_trees()

if __name__ == "__main__":
    # Print the trees for verification
    for tree_name, tree in TREES.items():
        print(f"\n{tree_name.capitalize()} Tree:")
        print(tree)

    # Save the trees to JSON files
    save_trees_to_json(TREES)
