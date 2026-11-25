#!/usr/bin/env python
"""Entry point for the model-plotting GUI.

Usage (in the ``mtinv`` environment; the ``mtplot`` command does the same):
    python run_plot.py
"""
import sys

from mtinv_gui.plotting.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
