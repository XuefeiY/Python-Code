"""
Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.
"""

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
    def __repr__(self):           # representation     
        if self:
            return "{} -> {}".format(self.val, self.next)
        
# My Solution
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        L1, L2 = [], []
        while l1:
            L1.append(l1.val)
            l1 = l1.next
        while l2:
            L2.append(l2.val)
            l2 = l2.next  
            
        L = sorted(L1+L2)
        
        def list_to_link(lst):
            """Takes a Python list and returns a Link with the same elements.
            """
            if len(lst) == 1:
                return ListNode(lst[0])
            new = ListNode(lst[0])
            new.next = list_to_link(lst[1:])
            return new
        return list_to_link(L)
 
# Test
l1 = ListNode(1)
l1.next = ListNode(3)
l1

l2 = ListNode(2)
l2.next = ListNode(4)
l2.next.next = ListNode(5)
l2

Solution().mergeTwoLists(l1, l2)
        
            
# Reference Solution
# Method 1: Using Dummy Nodes
"""
The strategy here uses a temporary dummy node as the start of the result list. The pointer Tail always points to the last node in the result list, so appending new nodes is easy.
The dummy node gives tail something to point to initially when the result list is empty. This dummy node is efficient, since it is only temporary, and it is allocated in the stack. 
The loop proceeds, removing one node from either ‘l1’ or ‘l2’, and adding it to tail. When we are done, the result is in dummy.next.

A dummy head node is a node that always points to the beginning of the list or has a NULL pointer field if the list is empty.
"""
class Solution(object):
    def mergeTwoLists1(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummyhead = ListNode(0)   
        lastNode = dummyhead
        while l1 and l2:
            if l1.val <= l2.val:
                lastNode.next = l1
                l1 = l1.next
            else:
                lastNode.next = l2
                l2 = l2.next
            lastNode = lastNode.next
            lastNode
        lastNode.next = l1 or l2
        return dummyhead.next
    
l1 = ListNode(1)
l1.next = ListNode(3)
l1

l2 = ListNode(2)
l2.next = ListNode(4)
l2.next.next = ListNode(5)
l2

Solution().mergeTwoLists1(l1, l2)

# Method 2: Using Recursion
"""
Merge is one of those nice recursive problems where the recursive solution code is much cleaner than the iterative code. 
You probably wouldn’t want to use the recursive version for production code however, 
because it will use stack space which is proportional to the length of the lists.
"""      
class Solution(object):
    def mergeTwoLists2(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if not l1 or not l2:
            return l1 or l2
        if l1.val <= l2.val:
            l1.next = self.mergeTwoLists2(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists2(l2.next, l1)
            return l2

# Test
l1 = ListNode(1)
l1.next = ListNode(3)
l1

l2 = ListNode(2)
l2.next = ListNode(4)
l2.next.next = ListNode(5)
l2

Solution().mergeTwoLists2(l1, l2)








