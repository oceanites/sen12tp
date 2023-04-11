format:
	black sen12tp/*.py tests/*.py

typechecking:
	#mypy sen12tp/dataset.py sen12tp/utils.py sen12tp/datamodule.py  sen12tp/constants.py
	mypy --python-executable venv/bin/python3.10 sen12tp/dataset.py sen12tp/utils.py sen12tp/datamodule.py  sen12tp/constants.py
