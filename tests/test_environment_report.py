import unittest

from protein_quanta.environment import build_environment_report


class _FakeCuda:
    def is_available(self):
        return True

    def device_count(self):
        return 1

    def get_device_name(self, index):
        return "NVIDIA Test GPU"

    def get_device_capability(self, index):
        return (8, 6)

    def get_device_properties(self, index):
        return type("Properties", (), {"total_memory": 48 * 1024**3})()


class _FakeTorch:
    __version__ = "2.6.0+cu124"
    version = type("Version", (), {"cuda": "12.4"})()
    cuda = _FakeCuda()


class EnvironmentReportTests(unittest.TestCase):
    def test_report_contains_only_reproducibility_whitelist(self):
        versions = {"numpy": "2.0.0", "missing": None}

        report = build_environment_report(
            package_versions=versions,
            python_version="3.12.3",
            system="Linux",
            release="6.8.0",
            machine="x86_64",
            torch_module=_FakeTorch(),
        )

        self.assertEqual(
            set(report), {"schema_version", "python", "platform", "packages", "torch"}
        )
        self.assertEqual(report["packages"], versions)
        self.assertNotIn("hostname", str(report).lower())
        self.assertNotIn("environment", str(report).lower())

    def test_cuda_device_metadata_is_normalized(self):
        report = build_environment_report(
            package_versions={},
            python_version="3.12.3",
            system="Linux",
            release="6.8.0",
            machine="x86_64",
            torch_module=_FakeTorch(),
        )

        self.assertEqual(report["torch"]["cuda_runtime"], "12.4")
        self.assertEqual(report["torch"]["device_count"], 1)
        self.assertEqual(report["torch"]["devices"][0]["compute_capability"], "8.6")
        self.assertEqual(report["torch"]["devices"][0]["memory_gib"], 48.0)

    def test_cpu_only_torch_does_not_enumerate_devices(self):
        torch_module = _FakeTorch()
        torch_module.cuda = type(
            "CpuCuda",
            (),
            {"is_available": lambda self: False, "device_count": lambda self: 0},
        )()

        report = build_environment_report(
            package_versions={},
            python_version="3.12.3",
            system="Windows",
            release="11",
            machine="AMD64",
            torch_module=torch_module,
        )

        self.assertFalse(report["torch"]["cuda_available"])
        self.assertEqual(report["torch"]["devices"], [])


if __name__ == "__main__":
    unittest.main()

