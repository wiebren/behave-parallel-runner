"""
Parallel test runner for `behave`_: runs feature files concurrently in
worker processes when ``--jobs N`` (N > 1) is used.

.. code-block:: ini

    # -- FILE: behave.ini
    [behave]
    runner = behave_parallel_runner:ParallelRunner

.. _behave: https://github.com/behave/behave
"""

from .runner import ParallelRunner

__all__ = ["ParallelRunner"]
__version__ = "0.1.0"
