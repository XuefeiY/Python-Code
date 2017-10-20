# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:53:01 2017

@author: xuefei.yang
"""

xs = [3,2,1]
ys = [1,2,3]
list(zip(xs,ys))
sorted(zip(xs,ys))
list(zip(sorted(zip(xs, ys))))
list(zip(*sorted(zip(xs, ys))))     # *-operator: sequence   **-operator: mapping


# =============================================================================
# Unpacking Argument Lists
# =============================================================================
#def func(first_param, *args, **kwargs):
#    Do something with the arguments here.
 
    
# Normal call with separate arguments
range(1,5)          # in python 3, range() returns a generator
list(range(1,5))    # convert it to a list


# Call with arguments unpacked from list
args = [1, 5]
range(*args)         
list(range(*args))



# Unpack the arguments from a dictionary
def make_sandwich(bread='brown', cheese='raw', lettuce='fresh', meat='extra'):
    print ('I am making a super tasty sandwich for you !')
    print ('You asked for %s bread, there you go !'%bread)
    print ('You shall have %s cheese, %s lettuce and %s meat.'%(cheese, lettuce, meat))

order = {'bread': 'parmesan-oregano', 'cheese': 'toasted', 'lettuce': 'fresh', 'meat': 'extra'}
make_sandwich(**order)    # We're passing an unpacked dictionary here.


# All positional arguments come before *args, and *args comes before **kwargs.