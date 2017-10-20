# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:39:57 2017

@author: xuefei.yang
"""
# Classes and Object Orientied Programming

# =============================================================================
# # Example 1: __init__
# =============================================================================
class Customer(object):
    """A customer of ABC Bank with a checking account. Customers have the
    following properties:

    Attributes:
        name: A string representing the customer's name.
        balance: A float tracking the current balance of the customer's account.
    """

    def __init__(self, name, balance=0.0):
        """Return a Customer object whose name is *name* and starting
        balance is *balance*."""
        self.name = name
        self.balance = balance

    def withdraw(self, amount):
        """Return the balance remaining after withdrawing *amount*
        dollars."""
        if amount > self.balance:
            raise RuntimeError('Amount greater than available balance.')
        self.balance -= amount
        return self.balance

    def deposit(self, amount):
        """Return the balance remaining after depositing *amount*
        dollars."""
        self.balance += amount
        return self.balance


# Remark:
# 1. self is the instance of the Customer that withdraw is being called on
# 2. After __init__, we consider the Customer object "initialized" and ready for use
# 3. The rule of thumb is, don't introduce a new attribute outside of the __init__ method, 
# otherwise you've given the caller an object that isn't fully initialized.

 
# =============================================================================
# # Example 2: main()
# =============================================================================
class AnimalActions:
    def quack(self): return self.strings['quack']
    def bark(self): return self.strings['bark']

class Duck(AnimalActions):
    strings = dict(
        quack = "Quaaaaak!",
        bark = "The duck cannot bark.",
    )

class Dog(AnimalActions):
    strings = dict(
        quack = "The dog cannot quack.",
        bark = "Arf!",
    )

def in_the_doghouse(dog):
    print(dog.bark())

def in_the_forest(duck):
    print(duck.quack())

def main():
    donald = Duck()
    fido = Dog()

    print("- In the forest:")
    for o in ( donald, fido ):
        in_the_forest(o)

    print("- In the doghouse:")
    for o in ( donald, fido ):
        in_the_doghouse(o)

if __name__ == "__main__": main()

# Summary: if __name__ == '__main__': has two primary use cases:
# 1. Allow a module to provide functionality for import into other code while also providing useful semantics as a standalone script 
# (a command line wrapper around the functionality)
# 2. Allow a module to define a suite of unit tests which are stored with (in the same file as) the code to be tested and which can be 
# executed independently of the rest of the codebase.