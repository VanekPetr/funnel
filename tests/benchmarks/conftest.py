"""Pytest configuration for benchmark tests.

This file can be used to add custom fixtures or configuration
for your benchmark tests.

Security Notes:
- S101 (assert usage): Asserts are appropriate in test code for validation.
- S603 (subprocess calls): Any subprocess usage in tests uses explicit argument lists, not shell=True.
- S607 (partial executable path): Known commands resolved from PATH are safe in test context.
"""
