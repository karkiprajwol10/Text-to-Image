import re

def contains_only_non_alphanumeric_chars(string):
    # Use a regular expression to check if the string contains only non-alphanumeric characters
    # Non-alphanumeric characters are defined as any character that is not a letter or a digit
    # The "^" character in the regular expression means "not"
    # The "$" character in the regular expression means "end of string"
    # The "*" character in the regular expression means "0 or more occurrences"
    # The "|" character in the regular expression means "or"
    if re.match("^[^a-zA-Z0-9@\$#]*$", string):
        return True
    else:
        return False
    




