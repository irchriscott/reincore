; ReinCore NSIS installer hooks
;
; Helpers for:
;   - PATH registration/unregistration of $INSTDIR
;   - Desktop-app optional section state (default checked)
;   - Shortcuts for the CLI (reincore / rc)

Var PathUpdated
Var InstalledDesktopApp

!macro AddToPath un
Function ${un}AddToPath
  ; Register $INSTDIR in user PATH
  ReadRegStr $0 HKCU "Environment" "Path"
  ${If} $0 == ""
    StrCpy $0 "$INSTDIR"
  ${Else}
    ; Delimit both sides: a similarly named directory is not this entry.
    StrCpy $1 ";$0;"
    ${WordReplace} "$1" ";$INSTDIR;" ";" "+" $2
    ${If} $1 == $2
      StrCpy $0 "$0;$INSTDIR"
    ${Else}
      Return
    ${EndIf}
  ${EndIf}
  WriteRegExpandStr HKCU "Environment" "Path" "$0"
  SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000
  StrCpy $PathUpdated 1
FunctionEnd
!macroend
!insertmacro AddToPath ""
!insertmacro AddToPath "un."

!macro RemoveFromPath un
Function ${un}RemoveFromPath
  ReadRegStr $0 HKCU "Environment" "Path"
  ${If} $0 == ""
    Return
  ${EndIf}
  StrCpy $2 ";$0;"
  ${WordReplace} "$2" ";$INSTDIR;" ";" "+" $1
  ${If} $1 == $2
    Return
  ${EndIf}
  ; Remove only complete PATH entries, preserving neighboring directories.
  ; Clean up leading/trailing/double semicolons
  Push "$1"
  Call ${un}TrimPath
  Pop $1
  WriteRegExpandStr HKCU "Environment" "Path" "$1"
  SendMessage ${HWND_BROADCAST} ${WM_SETTINGCHANGE} 0 "STR:Environment" /TIMEOUT=5000
  StrCpy $PathUpdated 1
FunctionEnd
!macroend
!insertmacro RemoveFromPath ""
!insertmacro RemoveFromPath "un."

; Simple in-string helper (case-sensitive). Pushes input then needle.
Function StrStr
  Exch $R1 ; needle
  Exch
  Exch $R2 ; haystack
  Push $R3
  Push $R4
  StrLen $R3 $R1
  StrCpy $R4 0
  loop:
    StrCpy $R5 "$R2" $R3 $R4
    StrCmp $R5 $R1 done
    StrCmp $R5 "" done
    IntOp $R4 $R4 + 1
    Goto loop
  done:
    StrCpy $R1 $R4
    Pop $R4
    Pop $R3
    Pop $R2
    Exch $R1
FunctionEnd

Function un.StrStr
  Exch $R1 ; needle
  Exch
  Exch $R2 ; haystack
  Push $R3
  Push $R4
  StrLen $R3 $R1
  StrCpy $R4 0
  loop:
    StrCpy $R5 "$R2" $R3 $R4
    StrCmp $R5 $R1 done
    StrCmp $R5 "" done
    IntOp $R4 $R4 + 1
    Goto loop
  done:
    StrCpy $R1 $R4
    Pop $R4
    Pop $R3
    Pop $R2
    Exch $R1
FunctionEnd

; Clean empty/duplicate semicolons from a path string
Function TrimPath
  Exch $R0
  Push $R1
  ; remove leading semicolons
  StrCpy $R1 "$R0" 1
  ${Do}
    ${If} $R1 == ";"
      StrCpy $R0 "$R0" "" 1
      StrCpy $R1 "$R0" 1
    ${Else}
      ${Break}
    ${EndIf}
  ${Loop}
  ; remove trailing semicolons
  StrLen $R1 $R0
  IntOp $R1 $R1 - 1
  StrCpy $R1 "$R0" 1 $R1
  ${Do}
    ${If} $R1 == ";"
      StrLen $R1 $R0
      IntOp $R1 $R1 - 1
      StrCpy $R0 "$R0" $R1
      StrLen $R1 $R0
      ${If} $R1 == 0
        ${Break}
      ${EndIf}
      IntOp $R1 $R1 - 1
      StrCpy $R1 "$R0" 1 $R1
    ${Else}
      ${Break}
    ${EndIf}
  ${Loop}
  ; collapse double semicolons
  StrCpy $R1 "$R0"
  loop_collapse:
    ${WordReplace} "$R1" ";;" ";" "+" $R0
    StrCmp $R0 $R1 done_collapse
    StrCpy $R1 "$R0"
    Goto loop_collapse
  done_collapse:
  Pop $R1
  Exch $R0
FunctionEnd

Function un.TrimPath
  Exch $R0
  Push $R1
  StrCpy $R1 "$R0" 1
  ${Do}
    ${If} $R1 == ";"
      StrCpy $R0 "$R0" "" 1
      StrCpy $R1 "$R0" 1
    ${Else}
      ${Break}
    ${EndIf}
  ${Loop}
  StrLen $R1 $R0
  IntOp $R1 $R1 - 1
  StrCpy $R1 "$R0" 1 $R1
  ${Do}
    ${If} $R1 == ";"
      StrLen $R1 $R0
      IntOp $R1 $R1 - 1
      StrCpy $R0 "$R0" $R1
      StrLen $R1 $R0
      ${If} $R1 == 0
        ${Break}
      ${EndIf}
      IntOp $R1 $R1 - 1
      StrCpy $R1 "$R0" 1 $R1
    ${Else}
      ${Break}
    ${EndIf}
  ${Loop}
  StrCpy $R1 "$R0"
  loop_collapse:
    ${WordReplace} "$R1" ";;" ";" "+" $R0
    StrCmp $R0 $R1 done_collapse
    StrCpy $R1 "$R0"
    Goto loop_collapse
  done_collapse:
  Pop $R1
  Exch $R0
FunctionEnd

; Create shortcuts for the CLI executables
Function CreateCLIShortcuts
  ${If} $NoShortcutMode = 1
    Return
  ${EndIf}
  !if "${STARTMENUFOLDER}" != ""
    CreateDirectory "$SMPROGRAMS\${STARTMENUFOLDER}\ReinCore CLI"
    CreateShortcut "$SMPROGRAMS\${STARTMENUFOLDER}\ReinCore CLI\Reincore.lnk" "$INSTDIR\reincore.exe" "--help"
    CreateShortcut "$SMPROGRAMS\${STARTMENUFOLDER}\ReinCore CLI\rc.lnk" "$INSTDIR\rc.exe" "--help"
  !else
    CreateDirectory "$SMPROGRAMS\ReinCore CLI"
    CreateShortcut "$SMPROGRAMS\ReinCore CLI\Reincore.lnk" "$INSTDIR\reincore.exe" "--help"
    CreateShortcut "$SMPROGRAMS\ReinCore CLI\rc.lnk" "$INSTDIR\rc.exe" "--help"
  !endif
FunctionEnd

Function un.RemoveCLIShortcuts
  ${If} $UpdateMode = 1
    Return
  ${EndIf}
  !if "${STARTMENUFOLDER}" != ""
    Delete "$SMPROGRAMS\$AppStartMenuFolder\ReinCore CLI\Reincore.lnk"
    Delete "$SMPROGRAMS\$AppStartMenuFolder\ReinCore CLI\rc.lnk"
    RMDir "$SMPROGRAMS\$AppStartMenuFolder\ReinCore CLI"
  !else
    Delete "$SMPROGRAMS\ReinCore CLI\Reincore.lnk"
    Delete "$SMPROGRAMS\ReinCore CLI\rc.lnk"
    RMDir "$SMPROGRAMS\ReinCore CLI"
  !endif
FunctionEnd
