import re
from collections import defaultdict



def parse_shakespeare_txt(filepath):
    # Hardcode the classification mapping based on MIT Shakespeare categories
    classification_map = {
        # Comedies
        "ALL’S WELL THAT ENDS WELL": "COMEDY",
        "AS YOU LIKE IT": "COMEDY",
        "THE COMEDY OF ERRORS": "COMEDY",
        "CYMBELINE": "COMEDY",
        "LOVE’S LABOUR’S LOST": "COMEDY",
        "MEASURE FOR MEASURE": "COMEDY",
        "THE MERCHANT OF VENICE": "COMEDY",
        "THE MERRY WIVES OF WINDSOR": "COMEDY",
        "A MIDSUMMER NIGHT’S DREAM": "COMEDY",
        "MUCH ADO ABOUT NOTHING": "COMEDY",
        "PERICLES, PRINCE OF TYRE": "COMEDY",
        "THE TAMING OF THE SHREW": "COMEDY",
        "THE TEMPEST": "COMEDY",
        "TWELFTH NIGHT; OR, WHAT YOU WILL": "COMEDY",
        "THE TWO GENTLEMEN OF VERONA": "COMEDY",
        "THE TWO NOBLE KINSMEN": "COMEDY",
        "THE WINTER’S TALE": "COMEDY",

        # Tragedies
        "THE TRAGEDY OF ANTONY AND CLEOPATRA": "TRAGEDY",
        "THE TRAGEDY OF CORIOLANUS": "TRAGEDY",
        "THE TRAGEDY OF HAMLET, PRINCE OF DENMARK": "TRAGEDY",
        "THE TRAGEDY OF JULIUS CAESAR": "TRAGEDY",
        "THE TRAGEDY OF KING LEAR": "TRAGEDY",
        "THE TRAGEDY OF MACBETH": "TRAGEDY",
        "THE TRAGEDY OF OTHELLO, THE MOOR OF VENICE": "TRAGEDY",
        "THE TRAGEDY OF ROMEO AND JULIET": "TRAGEDY",
        "THE LIFE OF TIMON OF ATHENS": "TRAGEDY",
        "THE TRAGEDY OF TITUS ANDRONICUS": "TRAGEDY",
        "TROILUS AND CRESSIDA": "TRAGEDY",

        # Histories
        "THE FIRST PART OF KING HENRY THE FOURTH": "HISTORY",
        "THE SECOND PART OF KING HENRY THE FOURTH": "HISTORY",
        "THE LIFE OF KING HENRY THE FIFTH": "HISTORY",
        "THE FIRST PART OF HENRY THE SIXTH": "HISTORY",
        "THE SECOND PART OF KING HENRY THE SIXTH": "HISTORY",
        "THE THIRD PART OF KING HENRY THE SIXTH": "HISTORY",
        "KING HENRY THE EIGHTH": "HISTORY",
        "THE LIFE AND DEATH OF KING JOHN": "HISTORY",
        "KING RICHARD THE SECOND": "HISTORY",
        "KING RICHARD THE THIRD": "HISTORY",

        # Poetry
        "A LOVER’S COMPLAINT": "POETRY",
        "THE PASSIONATE PILGRIM": "POETRY",
        "THE PHOENIX AND THE TURTLE": "POETRY",
        "THE RAPE OF LUCRECE": "POETRY",
        "VENUS AND ADONIS": "POETRY",

        # Sonnets
        "THE SONNETS": "SONNETS"
    }
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    content_by_category = defaultdict(list)
    full_corpus = []
    current_title = None
    buffer = []
    in_sonnets = False

    # Compile pattern to match sonnet numbers
    sonnet_number_pattern = re.compile(r'^\s*(\d{1,3})\s*$')
    known_titles = list(classification_map.keys())

    def flush_work():
        if not current_title or not buffer:
            return
        category = classification_map.get(current_title, "UNKNOWN")
        token = f"<<{category}>>"
        content = "".join(buffer).rstrip()
        section = f"{token}\n{current_title}\n{content}\n"
        full_corpus.append(section)
        content_by_category[category].append(section)

    i = 0
    parsing_started = False # Flag indicating if parsing is started => Used to skip the introductory text about gutenberg project
    parsing_end = False 
    seen_title_counts = {title: 0 for title in known_titles} # Keep track if we have seen any title a second time (to skip the Contents section)
    while i < len(lines) and not parsing_end:
        raw_line = lines[i]
        line = raw_line.strip('\n')


        # Marks the end of the real document. Rest is just Licence stuff
        if line.startswith('*** END OF THE PROJECT GUTENBERG EBOOK THE COMPLETE WORKS OF WILLIAM SHAKESPEARE ***'):
            parsing_end = True
            continue


        # Track known titles to skip TOC
        if not parsing_started and line.strip().isupper() and line.strip() in known_titles:
            seen_title_counts[line.strip()] += 1
            if seen_title_counts[line.strip()] == 2:
                # If this is a the second time we have seen a title start the actual parsing
                # This is the same block as down below for when we see a title, just that we set parsing_started = True
                parsing_started = True
                current_title = line.strip()
                buffer = []
                in_sonnets = (current_title == "THE SONNETS")
                i += 1
                # Preserve blank lines after title
                while i < len(lines) and lines[i].strip() == "":
                    buffer.append("\n")
                    i += 1
                continue
            else:
                i += 1
                continue

        if not parsing_started:
            i += 1
            continue

        # Detect SONNET sub-sections
        if in_sonnets:
            match = sonnet_number_pattern.match(line)
            if match:
                if len(buffer) > 1:
                    token = "<<SONNETS>>"
                    sonnet_title = buffer[0].strip()
                    section = f"{token}\n{sonnet_title}\n" + "".join(buffer[1:]).rstrip() + "\n" # Use rstrip() on buffer to get rid of the plank lines after a piece. Add one back as we still want a line break just not white lines 
                    full_corpus.append(section)
                    content_by_category["SONNETS"].append(section)
                buffer = [line.strip()]
                i += 1
                continue
            # Also flush final sonnet and then change in_sonnets to exit the subchapter
            if line.strip() == "THE END":
                in_sonnets = False
                # Same code as above (only that we set the current_title to None to not record anything 
                # that comes in between titles) => We do not need that anywhere else because we normally flush right before
                # a new chapter. This does not work here as we are currently in subchapters
                if len(buffer) > 1:
                    token = "<<SONNETS>>"
                    sonnet_title = buffer[0].strip()
                    section = f"{token}\n{sonnet_title}\n" + "".join(buffer[1:]).rstrip() + "\n" # Use rstrip() on buffer to get rid of the plank lines after a piece. Add one back as we still want a line break just not white lines 
                    full_corpus.append(section)
                    content_by_category["SONNETS"].append(section)
                    current_title = None
                buffer = [line.strip()]
                i += 1
                continue

            buffer.append(raw_line)
            i += 1
            continue

        # Check for new title
        if line.strip().isupper() and line.strip() in known_titles:
            flush_work()
            current_title = line.strip()
            buffer = []
            in_sonnets = (current_title == "THE SONNETS")
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                buffer.append("\n")
                i += 1
            continue

        buffer.append(raw_line)
        i += 1

    flush_work()  # flush last work

    # Compose the final strings
    result = {
        'full_corpus': "".join(full_corpus).strip(),
        'comedies_text': "".join(content_by_category["COMEDY"]).strip(),
        'tragedies_text': "".join(content_by_category["TRAGEDY"]).strip(),
        'histories_text': "".join(content_by_category["HISTORY"]).strip(),
        'poetry_text': "".join(content_by_category["POETRY"]).strip(),
        'sonnets_text': "".join(content_by_category["SONNETS"]).strip()
    }
    return result


if __name__ == '__main__':

    # Parse the whole shakespeare text
    parsed = parse_shakespeare_txt('Data/pg100.txt')

    # Print exemplary results
    print(parsed['full_corpus'][:1000])  # Preview full corpus
    print(parsed['tragedies_text'][:500])  # Preview only tragedies

    # Save the full corpus
    with open("Data/shakespeare_full_corpus.txt", "w", encoding="utf-8") as f:
        f.write(parsed['full_corpus'])

    # Save each category
    category_filenames = {
        "comedies_text": "Data/shakespeare_comedies.txt",
        "tragedies_text": "Data/shakespeare_tragedies.txt",
        "histories_text": "Data/shakespeare_histories.txt",
        "poetry_text": "Data/shakespeare_poetry.txt",
        "sonnets_text": "Data/shakespeare_sonnets.txt"
    }

    for category, filename in category_filenames.items():
        content = parsed.get(category, "")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
