"""
Python Basics — Environment Setup & Package Check (Starter File)

Complete assert_required_packages() so it asserts all required packages
are installed and importable on your machine.
"""

from __future__ import annotations

import importlib


REQUIRED_IMPORTS: dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "torch": "torch",
    "torchvision": "torchvision",
}


def assert_required_packages(required_imports: dict[str, str] = REQUIRED_IMPORTS) -> None:
    """
    Asserts all required packages can be imported.

    - `required_imports` maps a friendly package name -> import name.
      Example: "scikit-learn" -> "sklearn"
    """
    # TODO:
    # 1) Try importing each module in required_imports.values()
    # 2) Collect any that fail
    # 3) assert that none are missing (print a helpful message)
    raise NotImplementedError("TODO: implement assert_required_packages()")


def main() -> None:
    assert_required_packages()
    print("All required packages are installed and importable.")


if __name__ == "__main__":
    main()
