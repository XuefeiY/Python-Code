# List Comprehensions
S = [x**2 for x in range(10)]
V = [2**i for i in range(13)]
M = [x for x in S if x % 2 == 0]

print S; print V; print M

# List Comprehensions is a very powerful tool, which creates a new list based on another list, in a single, readable line.

# For example, let's say we need to create a list of integers which specify the length of each word in a certain sentence, but only if the word is not the word "the".
sentence = "the quick brown fox jumps over the lazy dog"
words = sentence.split()
word_lengths = []
for word in words:
      if word != "the":
          word_lengths.append(len(word))
print(words)
print(word_lengths)

# Using a list comprehension, we could simplify this process to this notation:
sentence = "the quick brown fox jumps over the lazy dog"
words = sentence.split()
word_lengths = [len(word) for word in words if word != "the"]
print(words)

# Using a list comprehension, create a new list called "newlist" out of the list "numbers", which contains only the positive numbers from the list, as integers.
numbers = [34.6, -203.4, 44.9, 68.3, -12.2, 44.6, 12.7]
newlist = [int(num) for num in numbers if num > 0]



# List Comprehensions & Lambda, map, filter and reduce
words = 'The quick brown fox jumps over the lazy dog'.split()
print words

stuff = [[w.upper(), w.lower(), len(w)] for w in words]
for i in stuff:
	print i

## map
stuff = map(lambda w: [w.upper(), w.lower(), len(w)], words)
for i in stuff:
	print i

## filter
stuff = filter(lambda w: len(w) % 2, words)
for i in stuff:
	print i

## reduce
reduce(lambda x,y: x+y, [47,11,42,13])

f = lambda a,b: a if (a > b) else b
reduce(f, [47,11,42,102,13])

reduce(lambda x, y: x+y, range(1,101))
