; Script generated by the Inno Setup Script Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

#define MyAppName "Avid Payment Predictor"
#define MyAppVersion "1.5"
#define MyAppPublisher "Willem van der Schans"
#define MyAppURL "https://github.com/Kydoimos97/PaymentPredictorUtility"
#define MyAppExeName "Avid Payment Predictor.exe"

[Setup]
; NOTE: The value of AppId uniquely identifies this application. Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{F4666FC9-23D8-4185-8402-0E4E85F4B686}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName=C:\Program Files\AvidPaymentPredictor
DisableProgramGroupPage=yes
LicenseFile=D:\Users\willem\OneDrive\MSIS\3. Fall 2022\Avid_Acceptance\AvidAcceptance\Github\PaymentPredictorUtility\ExternalFiles\License.txt.txt
InfoAfterFile=D:\Users\willem\OneDrive\MSIS\3. Fall 2022\Avid_Acceptance\AvidAcceptance\Github\PaymentPredictorUtility\ExternalFiles\InfoFile.txt
; Uncomment the following line to run in non administrative install mode (install for current user only.)
;PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=C:\Users\wille\Desktop\PaymentProb\CompilerOut
OutputBaseFilename=Avid Payment Predictor
SetupIconFile=D:\Users\willem\OneDrive\MSIS\3. Fall 2022\Avid_Acceptance\AvidAcceptance\Github\PaymentPredictorUtility\ExternalFiles\logoonly.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "C:\Users\wille\Desktop\PaymentProb\Output\Avid Payment Predictor\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\wille\Desktop\PaymentProb\Output\Avid Payment Predictor\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
