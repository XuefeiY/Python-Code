# =============================================================================
# List
# =============================================================================
a = [1,2,3]
b = [4,5,6]
a + b    # concatenate

a[0]
a[0] = 5
a

my_dict = {[a,b,c]:"some value"}  # TypeError: unhashable type: 'list'

# =============================================================================
# Tuple
# =============================================================================
c = (1,2,3)
d = (4,5,6)
c + d     # concatenate

c[0]
c[0] = 5  # error

my_dict = {("John","Wayne"): 90210}
my_dict
 
# Similarity:
# 1. Duplicates - Both tuples and lists allow for duplicates
# 2. Indexing, Selecting, & Slicing - Both tuples and lists index using integer values found within brackets. 
# 3. Comparing & Sorting - Two tuples or two lists are both compared by their first element, 
#    and if there is a tie, then by the second element, and so on. 
#    No further attention is paid to subsequent elements after earlier elements show a difference.

# Difference:
## By defination:
# 1. Syntax - Lists use [], tuples use ()
# 2. Mutability - Elements in a given list are mutable, elements in a given tuple are NOT mutable.
# 3. Hashtables (Dictionaries) - As hashtables (dictionaries) require that its keys are hashable and therefore immutable, 
#    only tuples can act as dictionary keys, not lists.
## By usage:
# 4. Homo vs. Heterogeneity of Elements - Tuples are heterogeneous data structures (i.e., their entries have different meanings), while lists are homogeneous sequences. 
# 5. Looping vs. Structures - Although both allow for looping, it only really makes sense to do it for a list. 
#    Tuples are more appropriate for structuring and presenting information 

# =============================================================================
# Set
# =============================================================================
# Sets are lists with no duplicate entries.
print(set("my name is Eric and Eric is my name".split()))

a = set(["Jake", "John", "Eric"])
print(a)
b = set(["John", "Jill"])
print(b)


print(a.intersection(b))
print(b.intersection(a))

print(a.symmetric_difference(b))
print(b.symmetric_difference(a))

print(a.difference(b))
print(b.difference(a))

print(a.union(b))


# =============================================================================
# Linked List
# =============================================================================
# Like arrays, Linked List is a linear data structure. Unlike arrays, linked list elements are not stored at contiguous location; 
# the elements are linked using pointers.
class node:
    def __init__(self):
        self.data = None # contains the data
        self.next = None # contains the reference to the next node


class linked_list:
    def __init__(self):
        self.cur_node = None

    def add_node(self, data):
        new_node = node() # create a new node
        new_node.data = data
        new_node.next = self.cur_node # link the new node to the 'previous' node.
        self.cur_node = new_node #  set the current node to the new one.

    def list_print(self):
        node = self.cur_node # cant point to ll!
        while node:
            print (node.data)
            node = node.next



ll = linked_list()
ll.add_node(1)
ll.add_node(2)
ll.add_node(3)

ll.list_print()


# =============================================================================
# Array
# =============================================================================
import numpy as np
e = np.array([1,2,3])
f = np.array([4,5,6])
e + f    # elementwise sum up

