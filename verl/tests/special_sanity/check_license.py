from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

license_head_bytedance = "Copyright 2024 Bytedance Ltd. and/or its affiliates"
license_head_bytedance_25 = "Copyright 2025 Bytedance Ltd. and/or its affiliates"
# Add custom license headers below
license_head_prime = "Copyright 2024 PRIME team and/or its affiliates"
license_head_individual = "Copyright 2025 Individual Contributor:"
license_head_sglang = "Copyright 2023-2024 SGLang Team"
license_head_modelbest = "Copyright 2025 ModelBest Inc. and/or its affiliates"
license_head_amazon = "Copyright 2025 Amazon.com Inc and/or its affiliates"
license_head_facebook = "Copyright (c) 2016-     Facebook, Inc"
license_headers = [
    license_head_bytedance,
    license_head_bytedance_25,
    license_head_prime,
    license_head_individual,
    license_head_sglang,
    license_head_modelbest,
    license_head_amazon,
    license_head_facebook,
]


def get_py_files(path_arg: Path) -> Iterable[Path]:
    """get py files under a dir. if already py file return it

    Args:
        path_arg (Path): path to scan for py files

    Returns:
        Iterable[Path]: list of py files
    """
    if path_arg.is_dir():
        return path_arg.glob("**/*.py")
    elif path_arg.is_file() and path_arg.suffix == ".py":
        return [path_arg]
    return []


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--directories",
        "-d",
        required=True,
        type=Path,
        nargs="+",
        help="List of directories to check for license headers",
    )
    args = parser.parse_args()

    # Collect all Python files from specified directories
    pathlist = set(path for path_arg in args.directories for path in get_py_files(path_arg))

    for path in pathlist:
        # because path is object not string
        path_in_str = str(path.absolute())
        print(path_in_str)
        with open(path_in_str, encoding="utf-8") as f:
            file_content = f.read()

            has_license = False
            for lh in license_headers:
                if lh in file_content:
                    has_license = True
                    break
            assert has_license, f"file {path_in_str} does not contain license"
