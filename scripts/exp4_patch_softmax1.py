import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to modeling_pythia.py")
    parser.add_argument("--backup", action="store_true")
    args = parser.parse_args()

    target = Path(args.file)
    text = target.read_text(encoding="utf-8")

    old = "attn_weights = torch.softmax(attn_weights, dim=-1)"
    new = (
        "attn_weights = torch.exp(attn_weights) / (torch.exp(attn_weights).sum(dim=-1, keepdim=True) + 1.0)"
    )

    if old not in text:
        raise RuntimeError("Pattern not found in file")

    if args.backup:
        backup = target.with_suffix(target.suffix + ".bak")
        backup.write_text(text, encoding="utf-8")

    target.write_text(text.replace(old, new), encoding="utf-8")
    print(f"Patched {target}")


if __name__ == "__main__":
    main()
