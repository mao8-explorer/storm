#!/bin/bash

# 一键安装 VSCode 插件到 Cursor IDE
# 使用方法：bash install_cursor_extensions.sh

EXTENSIONS=(
analytic-signal.preview-pdf
batisteo.vscode-django
bierner.markdown-checkbox
brandonfowler.exe-runner
cschlosser.doxdocgen
danielpinto8zz6.c-cpp-project-generator
docker.docker
docsmsft.docs-images
donjayamanne.githistory
donjayamanne.python-environment-manager
donjayamanne.python-extension-pack
dotjoshjohnson.xml
eamodio.gitlens
fittentech.fitten-code
formulahendry.code-runner
fougas.msys2
franneck94.c-cpp-runner
grapecity.gc-excelviewer
hoangkimlai.ipython
in4margaret.compareit
ionutvmi.path-autocomplete
jeff-hykin.better-cpp-syntax
leetcode.vscode-leetcode
lextudio.restructuredtext
mechatroner.rainbow-csv
mkxml.vscode-filesize
ms-azuretools.vscode-docker
ms-ceintl.vscode-language-pack-zh-hans
ms-dotnettools.csharp
ms-dotnettools.vscode-dotnet-runtime
ms-python.debugpy
ms-python.isort
ms-python.python
ms-python.vscode-pylance
ms-toolsai.jupyter
ms-toolsai.jupyter-keymap
ms-toolsai.jupyter-renderers
ms-toolsai.vscode-jupyter-cell-tags
ms-toolsai.vscode-jupyter-slideshow
ms-vscode-remote.remote-containers
ms-vscode-remote.remote-ssh
ms-vscode-remote.remote-ssh-edit
ms-vscode-remote.remote-wsl
ms-vscode-remote.vscode-remote-extensionpack
ms-vscode.cmake-tools
ms-vscode.cpptools
ms-vscode.cpptools-extension-pack
ms-vscode.cpptools-themes
ms-vscode.makefile-tools
ms-vscode.remote-explorer
ms-vscode.remote-server
ms-vsliveshare.vsliveshare
njpwerner.autodocstring
nvidia.isaacsim-vscode-edition
pascalreitermann93.vscode-yaml-sort
redhat.java
redhat.vscode-yaml
rimuruchan.vscode-fix-checksums-next
robodk.industrial-robots
s-nlf-fh.glassit
spywhere.guides
swyddfa.esbonio
trond-snekvik.simple-rst
twxs.cmake
vadimcn.vscode-lldb
visualstudioexptteam.intellicode-api-usage-examples
visualstudioexptteam.vscodeintellicode
vscjava.vscode-gradle
vscjava.vscode-java-debug
vscjava.vscode-java-dependency
vscjava.vscode-java-pack
vscjava.vscode-java-test
vscjava.vscode-maven
vscode-icons-team.vscode-icons
wayou.vscode-todo-highlight
yzhang.markdown-all-in-one
zhuangtongfa.material-theme
)

for EXT in "${EXTENSIONS[@]}"; do
    echo "Installing $EXT ..."
    cursor --install-extension "$EXT"
done

echo "✅ All extensions processed."
