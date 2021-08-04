"""
Change the plotting styles.
"""

import seaborn as sns


def use_style():
    """Set the default seaborn style to a predefined style that works well with the package."""

    sns.reset_defaults()
    sns.set_style("darkgrid")
    sns.set(font="Verdana")
    sns.set_context("paper")


def reset_style():
    """Restore the seaborn style to its defaults."""

    sns.reset_defaults()
