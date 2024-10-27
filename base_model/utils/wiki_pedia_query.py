import json
import wikipediaapi
import itertools
import re
from utils.data_processing import TREES  # Import the trees

# Initialize Wikipedia API
wiki_api = wikipediaapi.Wikipedia("base_model (sheldonandmore.com)", "en")


def fetch_category_members(details, depth, max_depth, visited):
    """Query and fetch category members for each category in the given details."""
    for category in details["categories"]:
        print(f"Fetching category members for category: {category}")
        category_page = wiki_api.page(f"Category:{category}")

        if category_page.exists():
            members = category_page.categorymembers

            # Store raw category member titles
            details["category_members"][category] = members

            # Recursively fetch details for each category member
            for member_title, member_obj in members.items():
                if member_title not in visited:
                    print(f"Fetching details for category member: {member_title}")
                    member_details = fetch_wikipedia_details(
                        member_title, depth + 1, max_depth, visited
                    )
                    if member_details:
                        details["category_members_query"][member_title] = member_details


def fetch_wikipedia_details(query, depth=0, max_depth=2, visited=None):
    """Fetch Wikipedia page details."""
    if visited is None:
        visited = set()

    if query in visited:
        print(f"Skipping already visited page: {query}")
        return None

    if depth > max_depth:
        print(f"Max depth reached for: {query}")
        return None

    visited.add(query)

    try:
        page = wiki_api.page(query)
        if not page.exists():
            print(f"No Wikipedia page found for '{query}'")
            return None

        details = {
            "page_title": page.title,
            "summary": page.summary[:2500],
            "full_url": page.fullurl,
            "sections": [
                {"title": section.title, "text": section.text[:2500]}
                for section in page.sections
            ],
            "links": list(page.links.keys()),
            "categories": [title for title in sorted(page.categories.keys())],
            "category_members": {},
            "category_members_query": {},
        }

        fetch_category_members(details, depth, max_depth, visited)

        return details

    except Exception as e:
        print(f"Error fetching data for '{query}': {e}")
        return None


def traverse_tree_and_fetch(tree):
    """Traverse the given tree and fetch Wikipedia details for each node."""
    print(f"Traversing tree: {tree.name}")
    details = fetch_wikipedia_details(tree.name)

    for child_name, child_node in tree.children.items():
        child_details = traverse_tree_and_fetch(child_node)
        if child_details:
            details["children"][child_name] = child_details

    return details


def query_and_enrich_trees():
    """Query Wikipedia for all trees in TREES."""
    enriched_data = {}

    for tree_name, tree in TREES.items():
        print(f"\nProcessing tree: {tree_name}")
        enriched_data[tree_name] = traverse_tree_and_fetch(tree)

    save_data_to_file(enriched_data, filename="enriched_data.json")


def generate_permutations(query):
    """Generate permutations for multi-word queries."""
    stop_words = {"and", "e.g.", "or", "of", "the"}
    words = [word.strip() for word in re.split(r"[(),]", query) if word.strip()]
    words = [word for word in words if word.lower() not in stop_words]

    if len(words) > 1:
        permutations = list(
            itertools.chain.from_iterable(
                itertools.permutations(words, r) for r in range(1, len(words) + 1)
            )
        )
        return [" ".join(p) for p in permutations]
    return [query]


def search_with_permutations(query):
    """Search Wikipedia using permutations if needed."""
    permutations = generate_permutations(query)
    for perm in permutations:
        details = fetch_wikipedia_details(perm)
        if details:
            return details
    return None


def save_data_to_file(data, filename="enriched_data.json"):
    """Save the enriched data to a file."""
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Data saved to {filename}")


def main():
    """Main function to query and enrich trees."""
    query_and_enrich_trees()


if __name__ == "__main__":
    main()
