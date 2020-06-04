Examples
--------
Lists several examples of Storchastic to help you get started. Also check out https://github.com/HEmile/storchastic/tree/master/examples.

The parrot module
=================

.. testsetup:: *

   import parrot

The parrot module is a module about parrots.

Doctest example:

.. doctest::

   >>> parrot.voom(3000)
   This parrot wouldn't voom if you put 3000 volts through it!

Test-Output example:

.. testcode::

   parrot.voom(3000)

This would output:

.. testoutput::

   This parrot wouldn't voom if you put 3000 volts through it!


.. toctree::
   :maxdepth: 3
   :caption: Examples:

   examples.discrete_vae