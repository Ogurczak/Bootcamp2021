#!/usr/bin/python3

import json
from argparse import ArgumentParser
import re
from copy import deepcopy
from pathlib import Path

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filepaths", type=Path, nargs="+")

    args = parser.parse_args()

    for filepath in args.filepaths:
        with open(filepath) as file:
            parsed = json.load(file)
            result = deepcopy(parsed)

            pattern = re.compile(r"(?<=>)\s*(\$.*?\$)\s*(?=<)")

            for i, cell in enumerate(parsed["cells"]):
                if cell["cell_type"] != "markdown":
                    continue

                new_source = []
                for i, line in enumerate(cell["source"]):
                    lines = re.split(pattern, line)
                    for i, line in enumerate(lines):
                        if line.startswith("$") and line.endswith("$"):
                            new_source.append("")
                            new_source.append(line)
                            new_source.append("")
                        else:
                            new_source.append(line)

                cell["source"] = new_source

        with open(filepath, "w") as file:
            json.dump(parsed, file,  indent=1)
