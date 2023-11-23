# Delcroix Thomas
# Lempereur Corentin
# Parent Anthony
# Blairon Mathis

import os
import re
import sys
import random
DAMPING = 0.85
SAMPLES = 10000


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    num_pages = len(corpus)
    prob_linked_page = damping_factor / len(corpus[page]) if len(corpus[page]) > 0 else 0

    prob_all_pages = (1 - damping_factor) / num_pages

    transition_probabilities = {p: prob_all_pages for p in corpus}

    for linked_page in corpus[page]:
        transition_probabilities[linked_page] += prob_linked_page

    return transition_probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = {page: 0 for page in corpus}
    sample = random.choice(list(corpus.keys()))

    for _ in range(n):
        page_rank[sample] += 1
        if random.random() < damping_factor and len(corpus[sample]) > 0:
            sample = random.choice(list(corpus[sample]))
        else:
            sample = random.choice(list(corpus.keys()))

    page_rank = {page: rank / n for page, rank in page_rank.items()}
    return page_rank


def iterate_pagerank(corpus, damping_factor):
    num_pages = len(corpus)
    page_rank = {page: 1 / num_pages for page in corpus}
    new_rank = page_rank.copy()

    change = True
    while change:
        change = False
        for page in page_rank:
            rank = (1 - damping_factor) / num_pages
            for other_page in corpus:
                if page in corpus[other_page]:
                    rank += damping_factor * page_rank[other_page] / len(corpus[other_page])
            new_rank[page] = rank

            if abs(new_rank[page] - page_rank[page]) > 0.001:
                change = True

        page_rank = new_rank.copy()

    return page_rank



if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")

    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")

    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")

    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
