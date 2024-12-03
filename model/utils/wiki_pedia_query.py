import json
import wikipediaapi
import itertools
import re
from model.utils.data_processing.data_pre_processing import TREES  # Import the trees

# Initialize Wikipedia API
wiki_api = wikipediaapi.Wikipedia("base_model (sheldonandmore.com)", "en")

log_data = {}


def log_and_print(stage, query, details):
    """Helper function to print and log details at each stage."""
    message = f"{stage} for '{query}': {details}"
    print(message)
    log_data.setdefault(stage, []).append(message)


def save_log_incrementally(stage, query, details, filename="enrichment_log.jsonl"):
    """Append each log entry to a file in JSON lines format."""
    log_entry = {stage: {query: details}}
    with open(filename, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"Log entry saved for {query} at stage {stage}")


def fetch_link_details(link_name):
    """Fetch details for each link."""
    link_page = wiki_api.page(link_name)
    if not link_page.exists():
        log_and_print("fetch_link_details", link_name, "No page found.")
        return None

    link_details = {
        "page_title": link_page.title,
        "summary": link_page.summary,
        "full_url": link_page.fullurl,
        "sections": [
            {"title": section.title, "text": section.text}
            for section in link_page.sections
        ],
        "links": list(link_page.links.keys())[:10],
        "categories": [title for title in sorted(link_page.categories.keys())],
    }
    log_and_print("fetch_link_details", link_name, link_details)
    save_log_incrementally("fetch_link_details", link_name, link_details)
    return link_details


def fetch_category_members(details, depth, max_depth, visited):
    initial_cat_members = []
    for category in details["categories"]:
        log_and_print("fetch_category_members", category, "Fetching members.")
        category_page = wiki_api.page(category)
        if category_page.exists():
            initial_cat_members.append(
                {"title": category_page.title, "ns": category_page.ns}
            )

    sub_members = get_category_members_recursive(
        [wiki_api.page(member["title"]) for member in initial_cat_members],
        level=0,
        max_level=1,
    )
    log_and_print("fetch_category_members", details["page_title"], sub_members)
    save_log_incrementally("fetch_category_members", details["page_title"], sub_members)
    return sub_members, initial_cat_members


def get_category_members_recursive(category_pages, level, max_level):
    members = {}
    for page in category_pages:
        members[page.title] = {"title": page.title, "ns": page.ns}
        if page.ns == wikipediaapi.Namespace.CATEGORY and level < max_level:
            sub_members = {}
            for sub_title, sub_page in page.categorymembers.items():
                if sub_page.ns == wikipediaapi.Namespace.CATEGORY:
                    sub_members[sub_title] = get_category_members_recursive(
                        [wiki_api.page(sub_title)], level=level + 1, max_level=max_level
                    )
            members[page.title]["category_members"] = sub_members
    log_and_print("final_category_sub_members", page.title, members)
    save_log_incrementally("final_category_sub_members", page.title, members)
    return members


def fetch_wikipedia_details(query, depth=0, max_depth=3000000, visited=None):
    if visited is None:
        visited = set()

    if query in visited or depth > max_depth:
        return None

    visited.add(query)
    details = None

    try:
        page = wiki_api.page(query)
        if not page.exists():
            log_and_print("fetch_wikipedia_details", query, "No page found.")
            return None

        details = {
            "page_title": page.title,
            "summary": page.summary,
            "full_url": page.fullurl,
            "sections": [
                {"title": section.title, "text": section.text}
                for section in page.sections
            ],
            "links": list(page.links.keys())[:10],
            "categories": [title for title in sorted(page.categories.keys())],
            "category_members": [],
            "category_sub_members": {},
        }
        log_and_print("initial_fetch", query, details)

        category_members, category_submembers = fetch_category_members(
            details, depth, max_depth, visited
        )
        details["category_members"] = category_members
        details["category_sub_members"] = category_submembers
        log_and_print("after_category_members", query, details)
        save_log_incrementally("initial_fetch", query, details)

    except Exception as e:
        log_and_print("Exception", query, str(e))
        save_log_incrementally("Exception", query, str(e))

    if details is not None:
        log_and_print("final_details", query, details)
        save_log_incrementally("final_details", query, details)
        return details

    return None


def generate_permutations(query):
    stop_words = {"and", "e.g.", "or", "of", "the"}
    words = [
        word.strip()
        for word in re.split(r"\s+", query)
        if word.lower() not in stop_words
    ]
    permutations = words[:]
    for i in range(2, len(words) + 1):
        for combo in itertools.permutations(words, i):
            permutations.append(" ".join(combo))
    return list(set(permutations))


def fetch_details_with_permutations(query, depth=0, max_depth=3000000, visited=None):
    api_query_child = {}
    original_details = fetch_wikipedia_details(query, depth, max_depth, visited)
    if original_details:
        api_query_child["original_query"] = original_details

    permutations = generate_permutations(query)
    for perm in permutations:
        if perm != query:
            perm_details = fetch_wikipedia_details(perm, depth, max_depth, visited)
            if perm_details:
                api_query_child[perm] = perm_details

    return api_query_child


def enrich_links(api_query_child):
    details = api_query_child.get("details", {})
    if "links" in details:
        details["link_details"] = {
            link: fetch_link_details(link) for link in details["links"]
        }
        log_and_print("after_link_details", details["page_title"], details)
        save_log_incrementally("after_link_details", details["page_title"], details)


def enrich_category_members(api_query_child):
    details = api_query_child.get("details", {})
    if "category_sub_members" in details:
        for category in details["category_sub_members"]:
            category_name = category["title"].replace("Category:", "")
            category_details = fetch_wikipedia_details(category_name)
            if category_details:
                category["details"] = category_details
                log_and_print(
                    "enrich_category_members", category_name, category_details
                )
                save_log_incrementally(
                    "enrich_category_members", category_name, category_details
                )


def traverse_and_enrich(node, depth=0, max_depth=3000000, visited=None):
    if visited is None:
        visited = set()
    print(f"Querying Wikipedia for node: {node.name}")
    log_data.setdefault("traverse_and_fetch_details", []).append(
        f"\nFetching details with permutations for node: {node.name}"
    )
    node.api_query_child = fetch_details_with_permutations(
        node.name, depth, max_depth, visited
    )

    save_enriched_tree(node, filename="intermediate_enriched_tree.json")
    save_log_to_json()

    for child in node.children.values():
        traverse_and_enrich(child, depth + 1, max_depth, visited)

    # Ensure `node` is returned at the end
    return node


def update_api_query_for_all_nodes(node, visited=None):
    if visited is None:
        visited = set()

    if node.name in visited:
        return
    print(f"enrichment query Wikipedia for node: {node.name}")
    visited.add(node.name)

    if node.api_query_child.get("details"):
        enrich_links(node.api_query_child)
        save_enriched_tree(node, filename="semi_final_enriched_tree.json")
        enrich_category_members(node.api_query_child)
        save_enriched_tree(node, filename="final_enriched_tree.json")
        save_log_to_json()

    for child in node.children.values():
        update_api_query_for_all_nodes(child, visited)


def save_enriched_tree(tree, filename="enriched_tree.json"):
    """Save the enriched tree to a JSON file."""
    with open(filename, "w") as f:
        json.dump(tree.to_dict(), f, indent=4)
    print(f"Enriched tree saved to {filename}")


def save_log_to_json(filename="enrichment_log.json"):
    """Save log data to JSON incrementally."""
    with open(filename, "w") as f:
        json.dump(log_data, f, indent=4)
    print(f"Log data saved to {filename}")


def load_and_build_enriched_trees(max_depth=3000000):
    """Build and enrich trees for each section, saving each tree individually."""
    enriched_trees = {}
    for tree_name, tree in TREES.items():
        print(f"\nEnriching tree: {tree_name}")
        # First pass: traverse and enrich the tree with initial details
        original_enriched_tree = traverse_and_enrich(
            tree, depth=0, max_depth=max_depth, visited=set()
        )

        if original_enriched_tree is None:
            print(f"Skipping tree '{tree_name}' due to enrichment failure.")
            continue

        save_enriched_tree(
            original_enriched_tree, filename=f"initial_enriched_batch_{tree_name}.json"
        )

        # Second pass: update API query for all nodes to enrich links and categories
        update_api_query_for_all_nodes(original_enriched_tree, visited=set())

        # Save each enriched tree after all enrichment is complete
        save_enriched_tree(
            original_enriched_tree,
            filename=f"final_batch_enriched_tree_{tree_name}.json",
        )

        enriched_trees[tree_name] = tree

    return enriched_trees


# Build the enriched tree and store in a global variable
ENRICHED_TREE = load_and_build_enriched_trees(max_depth=3000000)


def main():
    # Process each tree in TREES
    load_and_build_enriched_trees(max_depth=3000000)


if __name__ == "__main__":
    main()
