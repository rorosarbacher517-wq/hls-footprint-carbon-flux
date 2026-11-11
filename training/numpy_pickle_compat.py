"""Process-local NumPy module aliases for this project's legacy object NPY.

No package installation, input rewriting, np.load replacement or numeric
conversion. NumPy 2 pickles refer to numpy._core; old NumPy uses numpy.core.
This narrow bridge is validated against this dataset, not a promise that all
NumPy 2 pickles can be read by old versions.
"""
import importlib
import importlib.util
import sys


def install_numpy_pickle_compat():
    import numpy as np
    if 'numpy._core' in sys.modules or importlib.util.find_spec('numpy._core') is not None:
        return {'numpy_version': np.__version__, 'aliases_added': []}
    aliases = {
        'numpy._core': 'numpy.core',
        'numpy._core.multiarray': 'numpy.core.multiarray',
        'numpy._core.numeric': 'numpy.core.numeric',
    }
    modules = {new: importlib.import_module(old) for new, old in aliases.items()}
    for new, module in modules.items():
        sys.modules.setdefault(new, module)
    return {'numpy_version': np.__version__, 'aliases_added': list(modules)}


def run_script(script, arguments):
    """Bootstrap a pristine original runner after installing the same bridge."""
    from pathlib import Path
    import runpy
    info = install_numpy_pickle_compat()
    if info['aliases_added']:
        print('[NPY compatibility] process-local module aliases enabled; NumPy '+info['numpy_version'], flush=True)
    script = Path(script).resolve()
    sys.argv = [str(script)] + list(arguments)
    sys.path[0] = str(script.parent)
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: numpy_pickle_compat.py SCRIPT [original arguments...]')
    run_script(sys.argv[1], sys.argv[2:])
