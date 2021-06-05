import re

ALLOWED_CHARACTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZğĞüÜİışŞöÖçÇ "
lower_map = {ord(u"I"): u"ı", ord(u"İ"): u"i"}


def remove_punctuation(s):
    s = s.replace("→", " ")
    return re.sub(r"[!\"#$%&'()*+,-./:;<=>?@\[\\\]^_`{|}~]", " ", s)


def remove_repeating_spaces(s):
    return re.sub(r" +", " ", s).strip()


def filter_allowed_characters(s):
    return re.sub(r"[^{}]".format(ALLOWED_CHARACTERS), " ", s)


def to_lower(s):
    return s.translate(lower_map).lower()
