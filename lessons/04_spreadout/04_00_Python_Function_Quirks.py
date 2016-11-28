# -*- coding: utf8 -*-
from pylab import *
"""
###### Content under Creative Commons Attribution license CC-BY 4.0, code under MIT license (c)2015 G.F. Forsyth.
"""
"""
# Python Names
"""
"""
We introduced functions way back in Module 1 and they have served us well (and will continue to do so!).  There are a few behaviors that bear closer examination before we dive too deeply into the next lessons.  These are all common Python gotchas! that will continue to rear their heads, so it's better that you know about them now to avoid a large amount of hair-pulling when they crop up.  
"""
"""
There is a [fantastic talk](https://www.youtube.com/watch?v=_AEJHKGk9ns) from PyCon 2015 by Ned Batchelder on YouTube that will walk you through all of the things below and more, but we'll also include a short summary of some of the issues.
"""
"""
## Assigning variables
"""
"""
When we want a variable `x` to have a value of 5, we assign the *name* `x` to the *value* `5`.
"""
x = 5
print(x)
####################

"""
Integers are immutable.  They don't change.  If we assign another name `y` to be equal to `x`, there is no operation we can perform on `y` that will change `x` (or the 5).  
"""
y = x
print(y)
print(x)
####################

y += 1
print(y)
print(x)
####################

"""
Some datatypes *are* mutable, however, and that's where the trouble can start.  If instead of an integer, `x` points to a list (lists are mutable), then things are different.
"""
x = [1, 2, 3]
y = x
y.append(5)
print(x)
####################

"""
What happened?  We created a list `[1, 2, 3]` and pointed the name `x` at it.  Then we pointed the name `y` at the *same* list.  When we add a value to the list, there is only the one list, so the changes to it are reflected whether we ask for it by its first name, `x`, or its second name, `y`.
"""
"""
## What does this have to do with functions?
"""
"""
A reasonable question.  When you call a function and send it a few names (inputs), that action doesn't create copies of the objects that those names point to.  It just creates a *new* name that points at the *same* data.

Let's create a simple function that adds a value to a list and then returns a "copy" of that list.
"""
def add_to_list(mylist):
    mylist.append(7)
    
    newlist = mylist.copy()
    
    return newlist
####################

mylist = [1, 2, 3]
newlist = add_to_list(mylist)
####################

"""
We send in `mylist`, make a change to it, then make a copy of it and return the copy.  But we didn't return `mylist` so those changes are discarded, right?
"""
print(newlist)
####################

print(mylist)
####################

"""
Wrong.  We sent in the name `mylist` and then appended a value to it.  At that point, the list has been changed.  We used the `copy()` command to create `newlist`, so it points to a different list than `mylist`, but `mylist` has still been altered by the function.  
"""
"""
## What if we change the names?
"""
"""
Is this because the function expects a list named `mylist` and that is what we sent?  Alas, no.  
"""
T = [2, 4, 2]
newlist = add_to_list(T)
print(T)
####################

"""
When we send the name `T` to the function `add_to_list`, the function creates the new name `mylist` and points it to the same list that `T` points to.
"""
"""
## What do we do?
"""
"""
The most important thing is to be aware of this behavior.  It's a feature of the language and it doesn't often cause problems, but you need to know about it for when it does cause problems.  
"""
from IPython.core.display import HTML
css_file = '../../styles/numericalmoocstyle.css'
HTML(open(css_file, "r").read())
####################

print (" The presented result might be overlapping. ".center(60, "*"))
show()
