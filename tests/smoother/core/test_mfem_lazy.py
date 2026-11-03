"""The MFEM bridge is optional: everything imports, registers and documents without PyMFEM, and calling the
stage without it says how to install it.  Runs in subprocesses so a PyMFEM that *is* installed cannot leak in."""

import os
import subprocess
import sys
import textwrap

_HEX_TESTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "hex")


def _run(code):
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True)


def test_importing_the_smoothers_does_not_import_mfem_and_registers_tmop():
    out = _run("""
        import sys
        import nekmeshpy.quadmesh.smoother as q, nekmeshpy.hexmesh.smoother as h
        import nekmeshpy.core.smoother.mfem as m
        assert 'mfem' not in sys.modules, 'mfem was imported eagerly'
        assert 'tmop' in q.methods() and 'tmop' in h.methods()
        print('ok')
    """)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


def test_calling_tmop_without_mfem_names_the_install():
    out = _run("""
        import sys
        sys.modules['mfem'] = None            # as if not installed
        sys.path.insert(0, %r)
        from hexblocks import block
        from nekmeshpy.hexmesh import smoother as hs
        m = block(2)
        try:
            hs.tmop(m, hs.resolve(m, []))
        except ImportError as e:
            assert 'pip install mfem' in str(e), str(e)
            print('ok')
    """ % os.path.normpath(_HEX_TESTS))
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr
