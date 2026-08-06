"""Workers package.

The batch worker is run as its own process (``python -m
speechai.workers.batch_worker``); it is deliberately NOT imported here so that
``runpy`` does not double-import the module when the package initializes.
"""

__all__: list[str] = []
