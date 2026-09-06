#!/usr/bin/env bash
# Build su Windows/MSYS2 (gcc ucrt64 + mingw32-make). Su Linux basta `make`.
export PATH="/c/msys64/ucrt64/bin:$PATH"
T="$(cygpath -u "$LOCALAPPDATA" 2>/dev/null || echo "$HOME")/Temp"
export TMP="$T" TEMP="$T" TMPDIR="$T"
exec mingw32-make "$@"
