# install jupyterlab extensions
jupyter labextension install --no-build @krassowski/jupyterlab-lsp
jupyter labextension install --no-build @jupyterlab/debugger
jupyter lab build --dev-build=False --minimize=False