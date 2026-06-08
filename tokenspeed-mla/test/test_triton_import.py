import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


MLA_PYTHON = Path(__file__).resolve().parents[1] / "python"


def _run_triton_import(
    *, package: str | None = None, extra_pythonpath: str | None = None
) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [str(MLA_PYTHON)]
    if extra_pythonpath is not None:
        pythonpath_entries.insert(0, extra_pythonpath)
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    if package is None:
        env.pop("TOKENSPEED_TRITON_PACKAGE", None)
    else:
        env["TOKENSPEED_TRITON_PACKAGE"] = package

    code = textwrap.dedent(
        """
        import json
        import tokenspeed_mla._triton as tk_triton

        print(json.dumps({
            "triton_name": tk_triton.triton.__name__,
            "tl_name": tk_triton.tl.__name__,
        }, sort_keys=True))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def test_triton_import_defaults_to_tokenspeed_vendor_package():
    result = _run_triton_import()

    assert result == {
        "tl_name": "tokenspeed_triton.language",
        "triton_name": "tokenspeed_triton",
    }


def test_triton_import_can_use_plain_triton_package():
    custom_triton_pythonpath = os.environ.get("TOKENSPEED_TEST_TRITON_PYTHONPATH")
    if custom_triton_pythonpath is None:
        pytest.skip("set TOKENSPEED_TEST_TRITON_PYTHONPATH to a Triton source python path")

    if not Path(custom_triton_pythonpath, "triton", "__init__.py").is_file():
        pytest.fail(
            "TOKENSPEED_TEST_TRITON_PYTHONPATH must point at Triton's python directory"
        )

    result = _run_triton_import(
        package="triton", extra_pythonpath=custom_triton_pythonpath
    )

    assert result == {
        "tl_name": "triton.language",
        "triton_name": "triton",
    }
