"""
Resolve SemDeDup output paths to full image locations.

Input: text file with paths in format 'rel_tar_path|path' (e.g. split_000/00000.tar|00000/000000534.jpg)
Output: text file with full paths for tar extraction: '{dataset_path}/{rel_tar_path}::{path}'

Use these paths with tarfile to extract images: open tar at {dataset_path}/{rel_tar_path}, extract member 'path'.
"""

import os
import argparse

PATH_SEP = "|"
TAR_MEMBER_SEP = "::"


def main():
    parser = argparse.ArgumentParser(
        description="Resolve SemDeDup output paths to full image locations"
    )
    parser.add_argument(
        "input_txt",
        type=str,
        help="Input txt file with paths (rel_tar_path|path per line)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Base path to dataset (e.g. /datasets/.../laionasthetic_v2)",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default=None,
        help="Output txt file (default: input_txt with _resolved suffix)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["tar_member", "full_tar_path"],
        default="tar_member",
        help="Output format: 'tar_member' = {dataset_path}/{rel_tar}::{path} for tar extraction; "
        "'full_tar_path' = {dataset_path}/{rel_tar_path} only (tar path)",
    )
    args = parser.parse_args()

    with open(args.input_txt, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    resolved = []
    for line in lines:
        if PATH_SEP not in line:
            resolved.append(os.path.join(args.dataset_path, line))
            continue
        rel_tar_path, member_path = line.split(PATH_SEP, 1)
        full_tar_path = os.path.join(args.dataset_path, rel_tar_path)
        if args.format == "tar_member":
            resolved.append(f"{full_tar_path}{TAR_MEMBER_SEP}{member_path}")
        else:
            resolved.append(full_tar_path)

    output_path = args.output_txt or args.input_txt.replace(".txt", "_resolved.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(resolved))

    print(f"Resolved {len(resolved)} paths to {output_path}")
    if resolved:
        print(f"Sample: {resolved[0]}")


if __name__ == "__main__":
    main()
