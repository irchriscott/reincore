; Compile-only harness. Never execute the output: this validates hook syntax,
; macro expansion and variable/constant dependencies without touching user PATH.
Unicode true
Name "ReinCore hook syntax check"
OutFile "hooks-check.exe"
RequestExecutionLevel user
!include MUI2.nsh
!include WordFunc.nsh
!define STARTMENUFOLDER "ReinCore"
Var NoShortcutMode
Var UpdateMode
Var AppStartMenuFolder
!include "hooks.nsh"
Section
  Call AddToPath
  Call CreateCLIShortcuts
  WriteUninstaller "$TEMP\unused-reincore-hook-check.exe"
SectionEnd
Section "Uninstall"
  Call un.RemoveFromPath
  Call un.RemoveCLIShortcuts
SectionEnd
