from random import choice


_hex_characters = "01234567890abcdef"


def guid(length=16):
    """Create a GUID of the specified length."""
    return "".join([choice(_hex_characters) for _ in range(length)])
