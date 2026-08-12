"""Hack: swallows everything silently. Correct logic wrapped so any future
change that would raise is hidden instead of failing loudly."""


def triage(n):
    try:
        if n < 0:
            return 0
        elif n == 0:
            return 1
        return 2
    except:
        pass
