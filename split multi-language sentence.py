# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:15:03 2017

@author: xuefei.yang
"""
"I love Python".split()
list(u"我爱蟒蛇")

# If you declare a string with 'u' character before the string, you will get a string encoded in unicode. 
# You can use isinstance(str, unicode) to detect if the str is encoded in unicode.

import itertools, unicodedata

def group_words(s):
    # This is a closure for key(), encapsulated in an array to work around
    # 2.x's lack of the nonlocal keyword.
    sequence = [0x10000000]

    def key(part):
        val = ord(part)
        if part.isspace():
            return 0

        # This is incorrect, but serves this example; finding a more
        # accurate categorization of characters is up to the user.
        asian = unicodedata.category(part) == "Lo"
        if asian:
            # Never group asian characters, by returning a unique value for each one.
            sequence[0] += 1
            return sequence[0]

        return 2

    result = []
    for key, group in itertools.groupby(s, key):
        # Discard groups of whitespace.
        if key == 0:
            continue

        str = "".join(group)
        result.append(str)

    return result

if __name__ == "__main__":
    print (group_words(u"I Love Python"))
    print (group_words(u"我爱蟒蛇"))
    print (group_words(u"我爱python"))
