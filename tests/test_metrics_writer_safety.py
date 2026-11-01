from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import unittest

from utils.metrics import MetricsWriter


class MetricsWriterSafetyTests(unittest.TestCase):
    def test_metrics_header_failure_closes_new_file(self):
        stream = Mock()
        stream.write.side_effect = OSError("disk full")
        with tempfile.TemporaryDirectory() as directory, patch.object(Path, "open", return_value=stream):
            with self.assertRaises(OSError):
                MetricsWriter(Path(directory))
        stream.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
