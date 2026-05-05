# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the example-browser switch/reset args plumbing.

The companion :mod:`newton.tests.smoketest_example_browser` script runs every
registered example through the browser using a real GL viewer; it is not
auto-discovered.
"""

import unittest

import newton.examples
from newton.examples import _ExampleBrowser


class _StubViewer:
    """Minimal viewer stub for exercising _ExampleBrowser without a UI."""

    def __init__(self):
        self.cleared = 0

    def clear_model(self):
        self.cleared += 1


class _StubExample:
    """Captures the args namespace passed by the example browser."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=4)
        return parser


class TestExampleBrowserReset(unittest.TestCase):
    def test_reset_preserves_user_provided_args(self):
        # Simulate the user invoking the example with `--world-count 2`.
        args = _StubExample.create_parser().parse_args(["--world-count", "2"])
        self.assertEqual(args.world_count, 2)

        viewer = _StubViewer()
        browser = _ExampleBrowser(viewer, args)

        new_example = browser.reset(_StubExample)

        self.assertIsNotNone(new_example)
        self.assertEqual(new_example.args.world_count, 2)
        self.assertEqual(viewer.cleared, 1)

    def test_reset_falls_back_to_defaults_when_no_args(self):
        viewer = _StubViewer()
        browser = _ExampleBrowser(viewer)

        new_example = browser.reset(_StubExample)

        self.assertIsNotNone(new_example)
        self.assertEqual(new_example.args.world_count, 4)


if __name__ == "__main__":
    unittest.main()
