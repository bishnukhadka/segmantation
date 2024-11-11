from pathlib import Path

def check_suffix(file="deeplabv3plus.pt", suffix=(".pt",), msg=""):
    """Validates if a file or files have an acceptable suffix, raising an error if not."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"