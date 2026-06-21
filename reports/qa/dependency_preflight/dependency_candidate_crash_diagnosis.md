# Dependency Candidate Crash Diagnosis

The candidate environment has an import or binary compatibility failure before model loading: candidate_import_numeric_stack, candidate_import_pyarrow.

## candidate_import_numeric_stack

- Status: `CRASH`
- Return code: `-11`
- Signal: `SIGSEGV`
- Note: Isolates binary and import compatibility.

### Standard output

```text
(none)
```

### Standard error

```text
Fatal Python error: Segmentation fault

Current thread 0x00000001198b4e00 (most recent call first):
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 1176 in create_module
  File "<frozen importlib._bootstrap>", line 571 in module_from_spec
  File "<frozen importlib._bootstrap>", line 674 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pyarrow/__init__.py", line 65 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pandas/compat/pyarrow.py", line 8 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pandas/compat/__init__.py", line 27 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pandas/__init__.py", line 23 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "<string>", line 3 in <module>

Extension modules: numpy.core._multiarray_umath, numpy.core._multiarray_tests, numpy.linalg._umath_linalg, numpy.fft._pocketfft_internal, numpy.random._common, numpy.random.bit_generator, numpy.random._bounded_integers, numpy.random._mt19937, numpy.random.mtrand, numpy.random._philox, numpy.random._pcg64, numpy.random._sfc64, numpy.random._generator (total: 13)
```

## candidate_import_sklearn_joblib

- Status: `PASS`
- Return code: `0`
- Signal: `None`
- Note: Isolates binary and import compatibility.

### Standard output

```text
scikit-learn 1.5.0
joblib 1.3.2
```

### Standard error

```text
(none)
```

## candidate_import_xgboost

- Status: `PASS`
- Return code: `0`
- Signal: `None`
- Note: Isolates binary and import compatibility.

### Standard output

```text
xgboost 3.2.0
```

### Standard error

```text
(none)
```

## candidate_import_pyarrow

- Status: `CRASH`
- Return code: `-11`
- Signal: `SIGSEGV`
- Note: Isolates binary and import compatibility.

### Standard output

```text
(none)
```

### Standard error

```text
Fatal Python error: Segmentation fault

Current thread 0x0000000114404e00 (most recent call first):
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 1176 in create_module
  File "<frozen importlib._bootstrap>", line 571 in module_from_spec
  File "<frozen importlib._bootstrap>", line 674 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pyarrow/__init__.py", line 65 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "<string>", line 2 in <module>
```

## candidate_import_streamlit_pillow

- Status: `PASS`
- Return code: `0`
- Signal: `None`
- Note: Isolates binary and import compatibility.

### Standard output

```text
streamlit 1.54.0
pillow 12.2.0
image-class Image
```

### Standard error

```text
(none)
```

## metadata_load

- Status: `PASS`
- Return code: `0`
- Signal: `None`
- Note: Reads JSON only; no model deserialisation.

### Standard output

```text
metadata-loaded /Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/models/metadata/final_champion_metadata.json
feature-count 7
model-name None
```

### Standard error

```text
(none)
```

## current_environment_model_load

- Status: `PASS`
- Return code: `0`
- Signal: `None`
- Note: Confirms whether the existing model artifact is healthy in the current environment.

### Standard output

```text
loading-current-model /Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/models/trained/final_champion_model.joblib
current-model-type xgboost.sklearn XGBClassifier
```

### Standard error

```text
(none)
```

## candidate_environment_model_load

- Status: `PASS`
- Return code: `0`
- Signal: `None`
- Note: Isolates candidate-environment model deserialisation.

### Standard output

```text
loading-candidate-model /Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/models/trained/final_champion_model.joblib
candidate-model-type xgboost.sklearn XGBClassifier
```

### Standard error

```text
(none)
```

## candidate_environment_prediction

- Status: `CRASH`
- Return code: `-11`
- Signal: `SIGSEGV`
- Note: Runs only if the child process can load the model.

### Standard output

```text
(none)
```

### Standard error

```text
Fatal Python error: Segmentation fault

Current thread 0x0000000112d14e00 (most recent call first):
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 1176 in create_module
  File "<frozen importlib._bootstrap>", line 571 in module_from_spec
  File "<frozen importlib._bootstrap>", line 674 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pyarrow/__init__.py", line 65 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pandas/compat/pyarrow.py", line 8 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pandas/compat/__init__.py", line 27 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pandas/__init__.py", line 23 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "<string>", line 6 in <module>

Extension modules: numpy.core._multiarray_umath, numpy.core._multiarray_tests, numpy.linalg._umath_linalg, numpy.fft._pocketfft_internal, numpy.random._common, numpy.random.bit_generator, numpy.random._bounded_integers, numpy.random._mt19937, numpy.random.mtrand, numpy.random._philox, numpy.random._pcg64, numpy.random._sfc64, numpy.random._generator (total: 13)
```

## candidate_focused_tests

- Status: `CRASH`
- Return code: `-11`
- Signal: `SIGSEGV`
- Note: Excludes production-model loading until crash source is known.

### Standard output

```text
(none)
```

### Standard error

```text
Fatal Python error: Segmentation fault

Current thread 0x000000010e095e00 (most recent call first):
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 1176 in create_module
  File "<frozen importlib._bootstrap>", line 571 in module_from_spec
  File "<frozen importlib._bootstrap>", line 674 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pyarrow/__init__.py", line 65 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pandas/compat/pyarrow.py", line 8 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pandas/compat/__init__.py", line 27 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pandas/__init__.py", line 23 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/app/app_utils.py", line 23 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1078 in _handle_fromlist
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/tests/test_app_utils.py", line 5 in <module>
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/assertion/rewrite.py", line 188 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "<frozen importlib._bootstrap>", line 1050 in _gcd_import
  File "/Users/ruturajmokashi/.pyenv/versions/3.10.9/lib/python3.10/importlib/__init__.py", line 126 in import_module
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/pathlib.py", line 596 in import_path
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/python.py", line 508 in importtestmodule
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/python.py", line 561 in _getobj
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/python.py", line 290 in obj
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/python.py", line 577 in _register_setup_module_fixture
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/python.py", line 564 in collect
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/runner.py", line 406 in collect
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/runner.py", line 361 in from_call
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/runner.py", line 408 in pytest_make_collect_report
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pluggy/_callers.py", line 121 in _multicall
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pluggy/_manager.py", line 120 in _hookexec
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pluggy/_hooks.py", line 512 in __call__
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/runner.py", line 589 in collect_one_node
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/main.py", line 895 in _collect_one_node
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/main.py", line 1032 in genitems
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/main.py", line 869 in perform_collect
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/main.py", line 394 in pytest_collection
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pluggy/_callers.py", line 121 in _multicall
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pluggy/_manager.py", line 120 in _hookexec
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pluggy/_hooks.py", line 512 in __call__
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/main.py", line 383 in _main
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/main.py", line 330 in wrap_session
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/main.py", line 377 in pytest_cmdline_main
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pluggy/_callers.py", line 121 in _multicall
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pluggy/_manager.py", line 120 in _hookexec
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pluggy/_hooks.py", line 512 in __call__
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/config/__init__.py", line 229 in _main
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/_pytest/config/__init__.py", line 253 in _console_main
  File "/Users/ruturajmokashi/Ai Data Science Course Documents/Machine Learning/ECommerce_Conversion_Intelligence_Platform/.venv-security-upgrade/lib/python3.10/site-packages/pytest/__main__.py", line 9 in <module>
  File "/Users/ruturajmokashi/.pyenv/versions/3.10.9/lib/python3.10/runpy.py", line 86 in _run_code
  File "/Users/ruturajmokashi/.pyenv/versions/3.10.9/lib/python3.10/runpy.py", line 196 in _run_module_as_main

Extension modules: numpy.core._multiarray_umath, numpy.core._multiarray_tests, numpy.linalg._umath_linalg, numpy.fft._pocketfft_internal, numpy.random._common, numpy.random.bit_generator, numpy.random._bounded_integers, numpy.random._mt19937, numpy.random.mtrand, numpy.random._philox, numpy.random._pcg64, numpy.random._sfc64, numpy.random._generator (total: 13)
```

## candidate_main_pip_audit

- Status: `FAIL`
- Return code: `1`
- Signal: `None`
- Note: Checks the secure candidate requirement set.

### Standard output

```text
Name    Version ID             Fix Versions
------- ------- -------------- ------------
pyarrow 17.0.0  PYSEC-2026-113 23.0.1
```

### Standard error

```text
Found 1 known vulnerability in 1 package
```

## candidate_app_pip_audit

- Status: `FAIL`
- Return code: `1`
- Signal: `None`
- Note: Checks the secure candidate requirement set.

### Standard output

```text
Name    Version ID             Fix Versions
------- ------- -------------- ------------
pyarrow 17.0.0  PYSEC-2026-113 23.0.1
```

### Standard error

```text
Found 1 known vulnerability in 1 package
```

## candidate_streamlit_startup

- Status: `PASS`
- Return code: `0`
- Signal: `None`
- Note: Streamlit health endpoint returned HTTP 200.

### Standard output

```text
(none)
```

### Standard error

```text

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8765
  Network URL: http://192.168.178.25:8765
  External URL: http://62.216.204.252:8765

  For better performance, install the Watchdog module:

  $ xcode-select --install
  $ pip install watchdog

  Stopping...
```
